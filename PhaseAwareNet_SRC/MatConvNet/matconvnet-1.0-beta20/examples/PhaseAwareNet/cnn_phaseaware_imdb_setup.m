function imdb = cnn_phaseaware_imdb_setup( varargin )

opts.seed = 0;
opts.coverPath = 'C:\DeepLearning\matconvnet-1.0-beta20\data\JStego\75_mat';
opts.stegoPath = 'C:\DeepLearning\matconvnet-1.0-beta20\data\JStego\JUNI_0.4_mat';
opts.ratio = [0.6, 0.15, 0.25]; % train, validation, and test
opts.libSize = inf;
opts = vl_argparse( opts, varargin );

rng( opts.seed );
opts.ratio = opts.ratio/sum(opts.ratio);

% -------------------------------------------------------------------------
%                                                              Sanity Check
% -------------------------------------------------------------------------
fprintf('sanity check the library images ...') ;
targetSize = 10000;
expArray = linspace(1, targetSize, targetSize);

% first, sanilty the two data base
list = dir(fullfile(opts.coverPath, '*.mat'));
tokens = regexp({list.name}, '([\d]+).mat', 'tokens') ;
nameArray = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
if( ~isequal( sort(nameArray), expArray ) ) 
  error('coverPath = %s is corrupted', opts.coverPath);
end

list = dir(fullfile(opts.stegoPath, '*.mat'));
tokens = regexp({list.name}, '([\d]+).mat', 'tokens') ;
nameArray = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
if( ~isequal( sort(nameArray), expArray ) ) 
  error('stegoPath = %s is corrupted', opts.stegoPath);
end
fprintf('[checked]\n') ;

% meta
randomImages = randperm( targetSize );

totalSize = min( opts.libSize, targetSize );

numTrn = fix( totalSize * opts.ratio(1));
numVal = fix( totalSize * opts.ratio(2));
numTst = fix( totalSize * opts.ratio(3));

imdb.classes.name = {'Cover', 'Stego'} ;
n = strfind(opts.coverPath, filesep);
if( isempty( n ) )
  coverDes = 'Cover Images';
else
  coverDes = opts.coverPath(n(end)+1:end);
end

n = strfind(opts.stegoPath, filesep);
if( isempty( n ) )  
  stegoDes = 'Stego Images';
else
  stegoDes = opts.stegoPath(n(end)+1:end);
end

imdb.classes.description = {coverDes, stegoDes} ;
imdb.classes.coverPath = opts.coverPath;
imdb.classes.stegoPath = opts.stegoPath;

fprintf('%d Trn Image, %d Val Images, and %d Test Images \n ', ...
         numTrn, numVal, numTst) ;

% -------------------------------------------------------------------------
%                                                           Training images
% -------------------------------------------------------------------------
fprintf('searching training images ...') ;

names = cell(1, numTrn * 2 );
labels = ones(1, numTrn * 2 );              
for i = 1:numTrn
  
   idx = randomImages(i);
  
   names{2*i-1} = fullfile(opts.coverPath, strcat(num2str(idx),'.mat'));   
   labels(2*i - 1) = 1;
      
   names{2*i} = fullfile(opts.stegoPath, strcat(num2str(idx),'.mat'));
   labels(2*i) = 2;
end

imdb.images.id = 1:numel(names) ;
imdb.images.name = names ;
imdb.images.set = ones(1, numel(names)) ;
imdb.images.label = labels ;

fprintf('done\n') ;
% -------------------------------------------------------------------------
%                                                         Validation images
% -------------------------------------------------------------------------
fprintf('searching validation images ...') ;

names = cell(1, numVal * 2 );
labels = ones(1, numVal * 2 );

for i = 1:numVal
  
   idx = randomImages( numTrn + i);
  
   names{2*i-1} = fullfile(opts.coverPath, strcat(num2str(idx),'.mat'));   
   labels(2*i - 1) = 1;
      
   names{2*i} = fullfile(opts.stegoPath, strcat(num2str(idx),'.mat'));
   labels(2*i) = 2;

end

imdb.images.id  = horzcat( imdb.images.id, (1:numel(names)) + 1e7 - 1 );
imdb.images.name = horzcat(imdb.images.name, names );
imdb.images.set = horzcat( imdb.images.set, 2 * ones(1, numel(names)));
imdb.images.label = horzcat( imdb.images.label, labels ) ;

fprintf('done\n') ;

% -------------------------------------------------------------------------
%                                                               Test images
% -------------------------------------------------------------------------

fprintf('searching test images ...') ;

names = cell(1, numTst * 2 );
labels = ones(1, numTst * 2 );

for i = 1:numTst
  
   idx = randomImages( numTrn + numVal + i);
  
   names{2*i-1} = fullfile(opts.coverPath, strcat(num2str(idx),'.mat'));   
   labels(2*i - 1) = 1;
      
   names{2*i} = fullfile(opts.stegoPath, strcat(num2str(idx),'.mat'));
   labels(2*i) = 2;

end

imdb.images.id  = horzcat( imdb.images.id, (1:numel(names)) + 2e7 - 1 );
imdb.images.name = horzcat(imdb.images.name, names );
imdb.images.set = horzcat( imdb.images.set, 3 * ones(1, numel(names)));
imdb.images.label = horzcat( imdb.images.label, labels ) ;

fprintf('done\n') ;





