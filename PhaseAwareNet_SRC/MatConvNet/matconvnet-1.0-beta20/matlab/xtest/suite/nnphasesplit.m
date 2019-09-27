classdef nnphasesplit < nntest
  properties
    x
  end

  methods (TestClassSetup)
    function data(test,device)
      test.range = 10 ;
      test.x = test.randn(32,32,4,2) ;
      if strcmp(device,'gpu'), test.x = gpuArray(test.x) ; end
    end
  end

  methods (Test)
    function basic(test)
      x = test.x ;
      y = vl_nnphasesplit(x) ;
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnphasesplit(x,dzdy) ;
      test.der(@(x) vl_nnphasesplit(x), x, dzdy, dzdx, 1e-3) ;
    end
  end
end
