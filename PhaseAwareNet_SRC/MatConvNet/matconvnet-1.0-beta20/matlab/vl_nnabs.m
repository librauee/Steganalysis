function out = vl_nnabs(x,dzdy)
%VL_NNABS CNN ABS unit.
%   Y = VL_NNABS(X) computes the absolute value of the data X. X can
%   have an arbitrary size. The abs is defined as follows:
%
%     ABS(X) = |X|.
%
%   DZDX = VL_NNABS(X, DZDY) computes the derivative of the
%   block projected onto DZDY. DZDX and DZDY have the same
%   dimensions as X and Y respectively.

%   Note, MATLAB built-in function ABS() and SIGN() are used because their
%   support for gpuArray 

if nargin <= 1 || isempty(dzdy)
  out = abs( x ) ;
else
  out = dzdy .* sign( x );
end
