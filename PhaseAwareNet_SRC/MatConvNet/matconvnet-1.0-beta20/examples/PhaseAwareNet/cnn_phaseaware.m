function [net, info] = cnn_phaseaware(varargin)
%CNN_PHASEAWARE   Demonstrates training a PhaseAwareNet on JUNI and UED

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.modelType = 'PNet'; 
opts.seed = 0;   
opts.networkType = 'dagnn' ;
opts.batchSize = 40;
opts.lrSequence = 'log_short';
opts.printDotFile = true;
opts.coverPath = 'C:\DeepLearning\matconvnet-1.0-beta20\data\JStego\75_mat';
opts.stegoPath = 'C:\DeepLearning\matconvnet-1.0-beta20\data\JStego\JUNI_0.4_mat';


sfx = [opts.modelType, '-', opts.networkType, '-', num2str(opts.batchSize), ...
                  '-Seed-', num2str(opts.seed), '-', opts.lrSequence] ;   
opts.expDir = fullfile('data', ['JUNI-7504-' sfx]) ; % TODO

opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

opts.train = struct('gpus', [1,2], 'cudnn', true, 'stegoShuffle', true, 'computeBNMoment', true) ; 
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;


% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------
if (strcmpi( opts.modelType, 'PNet' ))                   
  
  net = cnn_phaseaware_PNet_init( 'networkType', opts.networkType, ...
                                 'batchSize', opts.batchSize, ...
                                 'seed', opts.seed, ...
                                 'lrSequence', opts.lrSequence ); 

elseif (strcmpi( opts.modelType, 'VNet' ))
  
  net = cnn_phaseaware_VNet_init( 'networkType', opts.networkType, ...
                                   'batchSize', opts.batchSize, ... 
                                   'seed', opts.seed, ...
                                   'lrSequence', opts.lrSequence ); 

else
  error('Unknown model type');
end

% put it to drawing
if ( ~exist( opts.expDir, 'dir' ) )
    mkdir( opts.expDir );  
end

if opts.printDotFile 
  net2dot(net,  fullfile( opts.expDir, 'NetConfig.dot' ), ...
          'BatchSize', net.meta.trainOpts.batchSize, ...
          'Inputs', {'input', [net.meta.inputSize, net.meta.trainOpts.batchSize]});
end

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------
if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = cnn_phaseaware_imdb_setup('coverPath', opts.coverPath, 'stegoPath', opts.stegoPath) ;

  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Set the class names in the network
net.meta.classes.name = imdb.classes.name ;
net.meta.classes.description = imdb.classes.description ;


% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------
switch opts.networkType
  case 'dagnn', trainFn = @cnn_train_dag ;
  otherwise, error('wrong network type');
end

[net, info] = trainFn(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      net.meta.trainOpts, ...
                      opts.train) ;

modelPath = fullfile(opts.expDir, 'net-deployed.mat');

switch opts.networkType
  case 'dagnn'
    net_ = net.saveobj() ;
    save(modelPath, '-struct', 'net_') ;
    clear net_ ;
end
                    
% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
bopts.useGpu = numel(opts.train.gpus) > 0 ;
bopts.imageSize = meta.inputSize;

switch lower(opts.networkType)
  case 'dagnn'
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end
            
% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
% label
labels = imdb.images.label(1,batch) ;
% images
images = zeros(opts.imageSize(1), opts.imageSize(2), ...
               opts.imageSize(3), numel(batch), 'single') ;
             
for i = 1:numel(batch)/2
  
%   cover = imread(imdb.images.name{batch(2*i-1)});
%   stego = imread(imdb.images.name{batch(2*i)});

  imt = load(imdb.images.name{batch(2*i-1)}, 'im');
  cover = single(imt.im);
    
  imt = load(imdb.images.name{batch(2*i)}, 'im');
  stego = single(imt.im);
  
  % random rotate, 0, 90, 180, 270  
  r = randi(4) - 1;  
  cover = rot90( cover, r );
  stego = rot90( stego, r );
  
  % random mirror flip  
  if ( rand > 0.5 )    
    cover = fliplr( cover );
    stego = fliplr( stego );
  end
  
  images(:,:,:,2*i-1) = single(cover);
  images(:,:,:,2*i) = single(stego);
   
end

if opts.useGpu > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;


