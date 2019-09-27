function [state,stats] = cnn_test_dag( imdb, getBatch, varargin)
%CNN_TEST_DAG Demonstrates test a CNN using the DagNN wrapper
%    CNN_TEST_DAG() is a slim version to CNN_TRAIN_DAG(), just do the
%    testing of the final net in the export 

opts.expDir = fullfile('data','exp') ;
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.test = [];
opts.gpus = [] ;
opts.prefetch = false ;
opts.testEpoch = inf;
opts.testSelect = [1, 1, 1]; % (1) training; (2)validation; (3), testing 
opts.saveResult = true;
opts.bnRefine = false;

opts.randomSeed = 0 ;
opts.stegoShuffle = false;
opts.cudnn = true ;
opts.extractStatsFn = @extractStats ;

opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isempty(opts.test), opts.test = find(imdb.images.set==3); end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

state.getBatch = getBatch ;

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

if ( opts.bnRefine )
  modelPath = @(ep) fullfile(opts.expDir, sprintf('bn-epoch-%d.mat', ep));
  resultPath  = @(ep) fullfile(opts.expDir, sprintf('test-bn-epoch-%d.mat', ep));
else
  modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
  resultPath  = @(ep) fullfile(opts.expDir, sprintf('test-net-epoch-%d.mat', ep));
end

start = findLastCheckpoint(opts.expDir) ;
if( start < 1 )
  error( 'Found no net' );
end

if start >= 1
  start = min(start, opts.testEpoch);
  fprintf('%s: testing by loading epoch name %s\n', mfilename, modelPath(start) );
  net = loadState(modelPath(start)) ;
end

% Make sure that we use the estimated BN moments
for i = 1:numel(net.layers)
  if ( isa( net.layers(i).block, 'dagnn.BatchNorm') )
    net.layers(i).block.computeMoment = false;
  end
end


for epoch = start
  
  % Set the random seed based on the epoch and opts.randomSeed.
  % This is important for reproducibility, including when training
  % is restarted from a checkpoint.
  
  rng(epoch + opts.randomSeed) ;
  prepareGPUs(opts, true ) ;

  % Train for one epoch.
  state.epoch = epoch ;
  
  % shuffle
  if( opts.stegoShuffle )
   
    N = numel(opts.train); % TRN
    M = numel(opts.val);   % VAL
    K = numel(opts.test);  % TST
    
    Lab = max( 1, numel(opts.gpus));
    
    % M and N must be even, and multiple Lab
    assert( ( rem( N, 2*Lab ) == 0 ) & ...
            ( rem( M, 2*Lab ) == 0 ) & ...
            ( rem( K, 2*Lab ) == 0 ) );
      
    seq = opts.train( 2*randperm(N/2) - 1 );
    seq = reshape( seq, Lab, N/(2*Lab) );
    state.train = reshape( [seq; seq+1], 1, N );
        
    seq = opts.val( 2*randperm(M/2) - 1 );
    seq = reshape( seq, Lab, M/(2*Lab) );
    state.val = reshape( [seq; seq+1], 1, M );
        
    seq = opts.test( 2*randperm(K/2) - 1 );
    seq = reshape( seq, Lab, K/(2*Lab) );
    state.test = reshape( [seq; seq+1], 1, K );
    
  else
    
    state.train = opts.train(randperm(numel(opts.train))) ; 
    state.val = opts.val(randperm(numel(opts.val))) ;
    state.test = opts.test(randperm(numel(opts.test))) ;

%     N = numel(opts.train); % TRN
%     M = numel(opts.val);   % VAL
%     K = numel(opts.test);  % TST
%     
% 
%     state.train = opts.train([1:2:N, 2:2:N]);
%     state.val = opts.val([1:2:M, 2:2:M]);
%     state.test = opts.test([1:2:K, 2:2:K]);

  end
  
  state.imdb = imdb ;

  if numel(opts.gpus) <= 1
    if( opts.testSelect(1) )
      stats.train = process_epoch(net, state, opts, 'train') ;
    end
    if( opts.testSelect(2) )
      stats.val   = process_epoch(net, state, opts, 'val') ;
    end
    if( opts.testSelect(3) )
      stats.test  = process_epoch(net, state, opts, 'test');
    end

  else
    savedNet = net.saveobj() ;
    spmd
      net_ = dagnn.DagNN.loadobj(savedNet) ;
      if( opts.testSelect(1) ) 
        stats_.train = process_epoch(net_, state, opts, 'train') ;
      end
      if( opts.testSelect(2) ) 
        stats_.val   = process_epoch(net_, state, opts, 'val') ;
      end
      if( opts.testSelect(3) )
        stats_.test  = process_epoch(net_, state, opts, 'test');
      end
      if labindex == 1, savedNet_ = net_.saveobj() ; end
    end
    net = dagnn.DagNN.loadobj(savedNet_{1}) ;
    stats__ = accumulateStats(stats_) ;
    
    if( opts.testSelect(1) )
      stats.train = stats__.train ;
    end
    if( opts.testSelect(2) )
      stats.val   = stats__.val ;
    end
    if( opts.testSelect(3) )
      stats.test  = stats__.test;
    end

    clear net_ stats_ stats__ savedNet savedNet_ ;
  end

  % save  
  if( opts.saveResult == true )
    saveState(resultPath(epoch), net, stats, state) ;
  end
 
end

% -------------------------------------------------------------------------
function stats = process_epoch(net, state, opts, mode)
% -------------------------------------------------------------------------

% move CNN  to GPU as needed
numGpus = numel(opts.gpus) ;
if numGpus >= 1
  net.move('gpu') ;
end

subset = state.(mode) ;
num = 0 ;
stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;
adjustTime = 0 ;

start = tic ;
for t=1:opts.batchSize:numel(subset)
  fprintf('%s: epoch %02d: %3d/%3d:', mode, state.epoch, ...
          fix((t-1)/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
  batchSize = min(opts.batchSize, numel(subset) - t + 1) ;

  for s=1:opts.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end

    inputs = state.getBatch(state.imdb, batch) ;

    if opts.prefetch
      if s == opts.numSubBatches
        batchStart = t + (labindex-1) + opts.batchSize ;
        batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
      state.getBatch(state.imdb, nextBatch) ;
    end
    
    net.mode = 'test' ;
    net.eval(inputs) ;
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
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------

for s = {'train', 'val', 'test'}
  s = char(s) ;
  total = 0 ;

  % initialize stats stucture with same fields and same order as
  % stats_{1}
  stats__ = stats_{1} ;
  if ( ~isfield(stats__, s) ) 
    continue; 
  end
  names = fieldnames(stats__.(s))' ;
  values = zeros(1, numel(names)) ;
  fields = cat(1, names, num2cell(values)) ;
  stats.(s) = struct(fields{:}) ;

  for g = 1:numel(stats_)
    stats__ = stats_{g} ;
    num__ = stats__.(s).num ;
    total = total + num__ ;

    for f = setdiff(fieldnames(stats__.(s))', 'num')
      f = char(f) ;
      stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;

      if g == numel(stats_)
        stats.(s).(f) = stats.(s).(f) / total ;
      end
    end
  end
  stats.(s).num = total ;
end

% -------------------------------------------------------------------------
function stats = extractStats(net)
% -------------------------------------------------------------------------
sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
stats = struct() ;
for i = 1:numel(sel)
  stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average ;
end

% -------------------------------------------------------------------------
function saveState(fileName, net, stats, state )
% -------------------------------------------------------------------------
net_ = net ;
net = net_.saveobj() ;
save(fileName, 'net', 'stats', 'state') ;

% -------------------------------------------------------------------------
function [net, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'stats') ;
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
