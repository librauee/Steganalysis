function [BN_Moments,stats] = cnn_bnrefine_dag( imdb, getBatch, varargin)
%CNN_TEST_DAG Demonstrates test a CNN using the DagNN wrapper
%    CNN_TEST_DAG() is a slim version to CNN_TRAIN_DAG(), just do the
%    testing of the final net in the export 

opts.expDir = fullfile('data','exp') ;
opts.batchSize = 256 ;
opts.train = [] ;
opts.val = [] ;
opts.test = [];
opts.gpus = [] ;
opts.prefetch = false ;
opts.testEpoch = inf;
opts.bnEpochCollectSize = 2000;
opts.saveResult = true;

opts.randomSeed = 0 ;
opts.stegoShuffle = false;
opts.cudnn = true ;
opts.extractStatsFn = @extractStats ;

opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isnan(opts.train), opts.train = [] ; end

% we must restrict the BN moment pooling from train set only

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

state.getBatch = getBatch ;

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
resultPath  = @(ep) fullfile(opts.expDir, sprintf('bn-epoch-%d.mat', ep));

start = findLastCheckpoint(opts.expDir) ;
if( start < 1 )
  error( 'Found no net' );
end

if start >= 1
  start = min(start, opts.testEpoch);
  fprintf('%s: testing by loading epoch %d\n', mfilename, start) ;
  net = loadState(modelPath(start)) ;
end

% First, create the structure to pool the BN moments
numLayers = numel(net.layers);

BN_Moments = struct('layer', {}, ...
                    'name', {}, ... 
                    'inputs', {}, ...
                    'outputs', {}, ...
                    'shape', {}, ...
                    'dataType', {}, ...
                    'oldValue', {}, ...
                    'hist', {} ) ;
                        
for i = 1:numLayers
  if ( isa( net.layers(i).block, 'dagnn.BatchNorm') )
    % Neet to save the BN moments for pooling
    net.layers(i).block.computeMoment = true;  
  
    name = net.layers(i).params{3};
    dataType = class(net.getParam(name).value);
    shape = size(net.getParam(name).value);
    
    BN_Moments(end+1).layer  = net.layers(i).name;
    BN_Moments(end).name     = name ;
    BN_Moments(end).inputs   = net.layers(i).inputs;
    BN_Moments(end).outputs  = net.layers(i).outputs; 
    BN_Moments(end).shape    = shape ;
    BN_Moments(end).dataType = dataType ;
    BN_Moments(end).oldValue = net.getParam(name).value;
  end
end


if( numel(opts.gpus) > 1 )
  error( 'cannot support multiple GPU now ')
end

numEpoch = ceil(opts.bnEpochCollectSize/(numel(opts.train)/opts.batchSize));
  
rng(start + opts.randomSeed) ;

for epoch = start:start + numEpoch - 1
  
  % Set the random seed based on the epoch and opts.randomSeed.
  % This is important for reproducibility, including when training
  % is restarted from a checkpoint.
  
  prepareGPUs( opts, true ) ;

  % Train for one epoch.
  state.epoch = epoch ;
  
  % shuffle
  if( opts.stegoShuffle )
   
    N = numel(opts.train); % TRN
    
    Lab = max( 1, numel(opts.gpus));
    
    % M and N must be even, and multiple Lab
    assert( rem( N, 2*Lab ) == 0 );
      
    seq = opts.train( 2*randperm(N/2) - 1 );
    seq = reshape( seq, Lab, N/(2*Lab) );
    state.train = reshape( [seq; seq+1], 1, N );
          
  else
    
    state.train = opts.train(randperm(numel(opts.train))) ;
  
  end
  
  state.imdb = imdb ;
  
  % keep pooling the result
  [stats.train(epoch - start + 1), BN_Moments] = process_epoch(net, state, opts, BN_Moments ) ;
  
end

% Reset the parameters
for i = 1:numel(BN_Moments)
  bn_moment_name = BN_Moments(i).name; 
  statsVal = median(BN_Moments(i).hist, 3);

  % set the new value
  paramIdx = net.getParamIndex(bn_moment_name);
  % double check the shape, see if it matches
  assert( isequal(size(statsVal), size(net.params(paramIdx).value ) ) );

  % reset the BN moment parameters
  net.params(paramIdx).value = statsVal;
end

% Revert it back
for i = 1:numel(net.layers)
  if ( isa( net.layers(i).block, 'dagnn.BatchNorm') )
     net.layers(i).block.computeMoment = false;
  end
end


saveState(resultPath(start), net, stats, BN_Moments ) ;

% -------------------------------------------------------------------------
function [stats, BN_Moments] = process_epoch(net, state, opts, BN_Moments )
% -------------------------------------------------------------------------

% move CNN  to GPU as needed
numGpus = numel(opts.gpus) ;
if numGpus >= 1
  net.move('gpu') ;
end

subset = state.train;
num = 0 ;
stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;
adjustTime = 0 ;

start = tic ;
for t=1:opts.batchSize:numel(subset)
  fprintf('%s: epoch %02d: %3d/%3d:', 'test', state.epoch, ...
          fix((t-1)/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
  batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
 
  % get this image batch and prefetch the next
  s = 1;
  batchStart = t + (labindex-1) + (s-1) * numlabs ;
  batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
  batch = subset(batchStart : numlabs : batchEnd) ;
  num = num + numel(batch) ;
  if numel(batch) == 0, continue ; end

  inputs = state.getBatch(state.imdb, batch) ;

  net.mode = 'test' ;

  net.eval(inputs) ;

  % update here
  for i = 1:numel(BN_Moments)
    layer_name = BN_Moments(i).layer;
    newVal = gather( net.getLayer(layer_name).block.moments );
    assert( ~isempty( newVal ) ); % in case the BatchNorm is not set up
    BN_Moments(i).hist = cat( 3,  BN_Moments(i).hist, newVal ); 
  end
    
  
  % get statistics
  time = toc(start) + adjustTime ;
  batchTime = time - stats.time ;
  stats = opts.extractStatsFn(net) ;
  stats.num = num ;
  stats.time = time ;
  currentSpeed = batchSize / batchTime ;
  averageSpeed = (t + batchSize - 1) / time ;
  if t == opts.batchSize + 1
    % compensate for the first iteration, which is an outlier
    adjustTime = 2*batchTime - time ;
    stats.time = time + adjustTime ;
  end

  fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
  for f = setdiff(fieldnames(stats)', {'num', 'time'})
    f = char(f) ;
    fprintf(' %s:', f) ;
    fprintf(' %.3f', stats.(f)) ;
  end
  fprintf('\n') ;
end

net.reset() ;
net.move('cpu') ;


% -------------------------------------------------------------------------
function stats = extractStats(net)
% -------------------------------------------------------------------------
sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
stats = struct() ;
for i = 1:numel(sel)
  stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average ;
end

% -------------------------------------------------------------------------
function saveState(fileName, net, stats, BN_Moments )
% -------------------------------------------------------------------------
net_ = net ;
net = net_.saveobj() ;
save(fileName, 'net', 'stats', 'BN_Moments') ;

% -------------------------------------------------------------------------
function net = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net' ) ;
net = dagnn.DagNN.loadobj(net) ;

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;


% -------------------------------------------------------------------------
function prepareGPUs(opts, cold)
% -------------------------------------------------------------------------
numGpus = numel(opts.gpus) ;
if numGpus > 1
  % check parallel pool integrity as it could have timed out
  pool = gcp('nocreate') ;
  if ~isempty(pool) && pool.NumWorkers ~= numGpus
    delete(pool) ;
  end
  pool = gcp('nocreate') ;
  if isempty(pool)
    parpool('local', numGpus) ;
    cold = true ;
  end
end
if numGpus >= 1 && cold
  fprintf('%s: resetting GPU\n', mfilename)
  if numGpus == 1
    gpuDevice(opts.gpus)
  else
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
end

%end
