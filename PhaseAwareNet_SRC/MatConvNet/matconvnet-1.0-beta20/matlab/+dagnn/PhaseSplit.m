classdef PhaseSplit < dagnn.Filter
  % Construct PhaseSplit in a way similar to Pooling, mightbe we could 
  % do it in a better and clean way. 
  properties
    poolSize = [1 1]
  end

  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nnphasesplit( inputs{1} ) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nnphasesplit( inputs{1}, derOutputs{1} ) ;
      derParams = {} ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      %outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(1) = inputSizes{1}(1)/8 ;
      outputSizes{1}(2) = inputSizes{1}(2)/8;
      outputSizes{1}(3) = inputSizes{1}(3)*64;
      outputSizes{1}(4) = inputSizes{1}(4);
    end

    function obj = PhaseSplit(varargin)
      %obj.load(varargin) ;
      obj.pad = [0 0 0 0];
      obj.stride = [8 8];
    end
  end
end
