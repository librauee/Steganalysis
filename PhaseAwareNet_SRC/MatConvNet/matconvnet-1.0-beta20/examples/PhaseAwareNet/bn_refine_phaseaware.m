function [BN_Moments, stats] = bn_refine_phaseaware(varargin)
% Test the phasesplit net

opts.batchSize = 48;
opts.expDir = fullfile('data', 'JUNI-7504-PNet-dagnn-40-Seed-0-log_short') ;
opts.testEpoch = 40;
opts.saveResult = true;
opts.bnEpochCollectSize = 2000;
opts.gpuIdx = 1;

opts = vl_argparse( opts, varargin );

opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

opts.train = struct('gpus', opts.gpuIdx, 'cudnn', true, 'stegoShuffle', true ) ; 
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% put it to drawing
if ( ~exist( opts.expDir, 'dir' ) )
    error('expDir is empty' ); 
end

% -------------------------------------------------------------------------
%                                                         Find the data base
% -------------------------------------------------------------------------
if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  error(' cannot find imdb' );
end


meta.inputSize = [512, 512, 1, opts.batchSize];

[BN_Moments, stats] = cnn_bnrefine_dag(imdb, getBatchFn( opts, meta ), ...
                      'expDir', opts.expDir, ...
                      'batchSize', opts.batchSize, ...
                      'testEpoch', opts.testEpoch, ...
                      'bnEpochCollectSize', opts.bnEpochCollectSize, ... 
                      'saveResult', opts.saveResult, ...
                      opts.train ) ;

                    
% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
bopts.useGpu = numel(opts.train.gpus) > 0 ;
bopts.imageSize = meta.inputSize;
    
fn = @(x,y) getDagNNBatch(bopts,x,y) ;
        
% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
% label
labels = imdb.images.label(1,batch) ;
% images
images = zeros(opts.imageSize(1), opts.imageSize(2), ...
               opts.imageSize(3), numel(batch), 'single') ;
for i = 1:numel(batch)
  imt = load(imdb.images.name{batch(i)}, 'im');
  images(:,:,:,i) = single(imt.im);
%   imt = imread(imdb.images.name{batch(i)});
%   images(:,:,:,i) = single(imt);
end

if opts.useGpu > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;
