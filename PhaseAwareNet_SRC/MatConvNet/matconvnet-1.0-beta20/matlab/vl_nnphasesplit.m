function y = vl_nnphasesplit( x, dzdy )
%VL_NNPHASESPLIT CNN phase split the feature plane into 8x8 = 64 DCT mode.
%   Y = VL_NNCROP(X, STRIDE) phase split the input X into 64 DCT phase mode. 
%
%   DZDX = VL_NNCROP(X, DZDY) computes the derivative DZDX of the
%   function projected on the output derivative DZDY. DZDX has the same
%   dimension as X and DZDY the same dimension as Y.
%

% dimension must be divided by 8
assert( rem(size(x,1), 8 ) == 0 & rem( size(x,2), 8) == 0 );

% Initialize some parameters
inputSize  = [size(x,1) size(x,2) size(x,3) size(x,4)] ;
outputSize = [size(x,1)/8, size(x,2)/8, 64*size(x,3), size(x,4)];

% zig zag order
zzag = zeros(64, 4);
idx = 1;
startCh = 1;
for i = 0:7
  for j = 0:7    
    stopCh = startCh + inputSize(3);
    zzag(idx, :) = [ i, j, startCh, stopCh - 1 ];
    idx = idx + 1;
    startCh = stopCh;
  end
end

% sampling array
sy = 1:8:inputSize(1); 
sx = 1:8:inputSize(2);
  
if nargin <= 1 || isempty(dzdy)  
  % forward function
  if isa( x, 'gpuArray' )
    y = gpuArray.zeros(outputSize, classUnderlying(x)) ;
  else
    y = zeros(outputSize, 'like', x ) ;
  end 
 
  for i = 1:64
    y(:,:,zzag(i,3):zzag(i,4),:) = x(sy + zzag(i,1), sx + zzag(i,2), :, : );   
  end
  
else
  % backward function
  if isa(dzdy, 'gpuArray')
    y = gpuArray.zeros(inputSize, classUnderlying(dzdy)) ;
  else
    y = zeros(inputSize, 'like', x) ;
  end
 
  for i = 1:64
    y(sy + zzag(i,1), sx + zzag(i,2), :, : ) = dzdy(:,:,zzag(i,3):zzag(i,4),:);   
  end
  
end
