local CrossInputNeighborhood, parent = torch.class('nn.CrossInputNeighborhood', 'nn.Module');

require 'io'
require 'cutorch'

function CrossInputNeighborhood:__init()
    self.output1 = torch.CudaTensor() 
    self.output2 = torch.CudaTensor()
    self.gradInput = torch.CudaTensor();
    
    --temporary variables
    self.gradInput1 = torch.CudaTensor();
    self.gradInput2 = torch.CudaTensor();
    self.gradOutput1 = torch.CudaTensor();
    self.gradOutput2 = torch.CudaTensor();
    
    parent.__init(self)
end


--[[
   
   name: updateOutput
   @param input - 50 layers of 12 x 37 patches
   @return - output of 2 tensors of size (625 x 12 x 37)
   
]]--
-- override the predefined methods
function CrossInputNeighborhood:updateOutput(input)
    ------------------------------------------------------------------------------
    -- the implementation should be done as below:
    --  input will contain X layers of MxN
    --  f_i = first X/2 layers
    --  g_i = second X/2 layers
    --  
    --  1) K1_i:
    --     from f_i, take every pixel neighborhood of 5x5, subtract the same pixel neighborhood of 5x5 from 
    --     g_i
    --     
    --  2) K2_i:
    --     from g_i, take every pixel neighborhood of 5x5, subtract the same pixel neighborhood of 5x5 from 
    --     f_i
    --     
    --  the output should contain totally X layers of MxNx5x5.
    --  
    
    input.nn.CrossInputNeighborhood_updateOutput(self, input);
    self.output = {self.output1, self.output2};
    
    return self.output; 
end

function CrossInputNeighborhood:updateGradInput(input, gradOutput)
    self.gradOutput1:resizeAs(gradOutput[1]):copy(gradOutput[1]);
    self.gradOutput2:resizeAs(gradOutput[2]):copy(gradOutput[2]);
    
    input.nn.CrossInputNeighborhood_updateGradInput(self, input);
    
    return self.gradInput
end 
