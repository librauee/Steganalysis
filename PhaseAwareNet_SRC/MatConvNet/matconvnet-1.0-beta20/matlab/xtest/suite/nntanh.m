classdef nntanh < nntest
  properties
    x
    delta
  end

  methods (TestClassSetup)
    function data(test,device)
      test.range = 10 ;
      test.x = test.randn(15,14,3,2) ;
      if strcmp(device,'gpu'), test.x = gpuArray(test.x) ; end
    end
  end

  methods (Test)
    function basic(test)
      x = test.x ;
      y = vl_nntanh(x) ;
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nntanh(x,dzdy) ;
      test.der(@(x) vl_nntanh(x), x, dzdy, dzdx, 1e-2) ;
    end
  end
end
