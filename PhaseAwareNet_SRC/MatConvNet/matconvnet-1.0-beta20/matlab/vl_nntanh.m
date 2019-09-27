function out = vl_nntanh(x,dzdy)
%VL_NNTANH CNN TanH hyperbolic non-linearity
%   Y = VL_NNTANH(X) computes the hyperbolic tangent non-linearity of the
%   data X. X can have an arbitrary size. The tanh is defined as follows:
%
%     TANH(X) = (EXP(2X) - 1 )/( EXP(2x) + 1 ).
%
%   DZDX = VL_NNTANH(X, DZDY) computes the derivative of the
%   block projected onto DZDY. DZDX and DZDY have the same
%   dimensions as X and Y respectively.
%
%   NOTE: Matlab build-in function TANH() is used since it has extended
%   support for gpuArray

y = tanh( x );

if nargin <= 1 || isempty(dzdy)
  out = y;
else
  out = dzdy .* ( 1 - y.*y );
end
