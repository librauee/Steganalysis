classdef nnabs < nntest
  properties
    x
    delta
  end

  methods (TestClassSetup)
    function data(test,device)
      % make sure that all elements in x are differentiable. in this way,
      % we can compute numerical derivatives reliably by adding a delta < .5.
      delta = 0.01;      
      test.range = 10 ;
      x = test.randn(15,14,3,2) ;
      
      ind = find(( x < 0 )&( x > -2*delta));
      if (~isempty(ind))
        x(ind) = -2 + rand([1, length(ind)], 'like', x);
      end
      
      test.x = x ;
      test.delta = delta;

      if strcmp(device,'gpu'), test.x = gpuArray(test.x) ; end
    end
  end

  methods (Test)
    function basic(test)
      x = test.x ;
      y = vl_nnabs(x) ;
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnabs(x,dzdy) ;
      test.der(@(x) vl_nnabs(x), x, dzdy, dzdx, 1e-2) ;
    end
  end
end
