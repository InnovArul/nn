
require 'io';

local CrossInputNeighborhood = torch.class('nn.CrossInputNeighborhood', 'nn.Module');


--[[
   
   name: allocateTensor
   @param
   @return allocates the tensor of the given size in GPU or CPU based on the option set
   
]]--
function allocateTensor(givenSize)
  opt.useCuda = true;
  
  if(opt.useCuda) then
     newTensor = torch.CudaTensor(unpack(givenSize));
     newTensor:zero()
  else
     newTensor = torch.Tensor(unpack(givenSize));
  end
  
  return newTensor;
end

--[[
   
   name: localizeMemory
   @param
   @return copies the given tensor to GPU, incase GPU usage is forced
   
]]--
function localizeMemory(tensor)
    -- hard coded
    opt.useCuda = true;

  if(opt.useCuda) then
     tensor:float();
     newTensor = tensor:cuda();
     newTensor:zero()
  else
    newTensor = tensor;
  end
  
  return newTensor;
end


--[[
   
   name: _getNeighborHood
   @param
   @return 5x5 neighborhood of the given pixel
   
]]--
function _getNeighborHood(frame, x, y)
    neighborhood = allocateTensor({25});
    framesize = frame:size();

    neighborhood:zero();
    for i = -2, 2 do
        for j = -2, 2 do
            if((x + i) > 0 and (y + j) > 0  and (x + i) <= framesize[1] and (y + j) <= framesize[2]) then
                neighborhood[((i + 2) * 5) + (j + 2) + 1] = frame[i + x][j + y];
            end
        end
    end
    
    return neighborhood;
end

--[[
   
   name: _getGradient
   @param gradOutput - gradOutput from the next layer w.r.t the final output
          x, y       - position of the element to which the gradient to be found
          layerIndex - layer index which has to be considered
   @return get the gradient of affected points in g_i for f_i & vice versa
   
]]--
function _getGradient(gradOutput, x, y, layerIndex)
    gradOutputSize = gradOutput:size()

    gradient = 0;
    position = 25;
    
    for i = x - 2, x + 2 do
        for j = y - 2, y + 2 do
            if(i >= 1 and j >= 1 and i <= gradOutputSize[2] and j <= gradOutputSize[3]) then
                -- take the particular gradient and add it to gradient variable
                layerOffset = (layerIndex - 1) * 25 + 1;
                elementOffset = position - 1;
                
                --NOTE: here multiply by -1 for the second part of the gradient (i.e., g_i - f_i)
                gradient = gradient + (gradOutput[layerOffset + elementOffset][i][j] * -1);
            end        
            
            position = position - 1;
        end
    end
    
    return gradient;
end

--[[
   
   name: _getCrossNeighborDifferences
   @param
   @return
   
]]--
function _getCrossNeighborDifferences(input, frame1index, frame2index, halfmaps)
    frame1 = input[frame1index];
    frame2 = input[frame2index];

    framesize = frame1:size();  
    
    Frame1toFrame2 = allocateTensor({25, framesize[1], framesize[2]});
    Frame2toFrame1 = allocateTensor({25, framesize[1], framesize[2]});
    
    for index1 = 1, framesize[1] do
        for index2 = 1, framesize[2] do
            -- calculate Fi - Gi
            -- get f_i * 1(5, 5)
            frame1Neighborhood = allocateTensor({5, 5}):fill(frame1[index1][index2]);
            --get the neighborhood of g_i       
            frame2Neighborhood = _getNeighborHood(frame2, index1, index2);
            frame1Toframe2Diff = frame1Neighborhood - frame2Neighborhood;           
            Frame1toFrame2[{{1, 25 },{index1},{index2}}] = frame1Toframe2Diff;
                        
            -- calculate Gi - Fi
            -- get the neighborhood of f_i
            frame1Neighborhood = _getNeighborHood(frame1, index1, index2);
            -- get g_i * 1(5, 5)
            frame2Neighborhood = allocateTensor({5, 5}):fill(frame2[index1][index2]);
            frame2Toframe1Diff = frame2Neighborhood - frame1Neighborhood;           
            Frame2toFrame1[{{1, 25 },{index1},{index2}}] = frame2Toframe1Diff;
        end
    end
    
    return Frame1toFrame2, Frame2toFrame1;
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
    
   -- print('inside custom layer updateOutput!')

    inputSize = input:size();
    
    -- 50 maps of cross input neighborhood differences
    local output1 = allocateTensor({inputSize[1]/2 * 5 * 5, inputSize[2], inputSize[3]});
    local output2 = allocateTensor({inputSize[1]/2 * 5 * 5, inputSize[2], inputSize[3]});

    mapIndex = 1;
    halfmaps = inputSize[1] / 2;
    for index = 1, halfmaps do
        Frame1toFrame2diff, Frame2toFrame1diff = _getCrossNeighborDifferences(input, index, halfmaps + index, halfmaps);
        output1[{{(index - 1) * 25 + 1, index * 25},{},{}}] = Frame1toFrame2diff;
        output2[{{(index - 1) * 25 + 1, index * 25},{},{}}] = Frame2toFrame1diff;
    end

    self.output = {output1, output2};
    --print('outputs')
    --print(output1)
    --print(output2)
    return self.output;
end

function CrossInputNeighborhood:updateGradInput(input, gradOutput)
  --print('inside custom layer updateGradInput!')
    local gradOutput1 = allocateTensor({})
    local gradOutput2 = allocateTensor({})
    gradOutput1:resizeAs(gradOutput[1]):copy(gradOutput[1]);
    gradOutput2:resizeAs(gradOutput[2]):copy(gradOutput[2]);

--[[
--DISPLAYING THE GRAD OUTPUT
    print(gradOutput1[{{1,3},{},{}}]);
    image.display(gradOutput1[{{1,3},{},{}}] * 255);
    image.display(gradOutput2[{{1,3},{},{}}] * 255);
    io.read();
    --]]
    
    gradOutputSize = gradOutput1:size();
    inputSize = input:size ();
    
    self.gradInput = allocateTensor({inputSize[1], gradOutputSize[2], gradOutputSize[3]});
    buffGradInput1 = allocateTensor({inputSize[1]/2, gradOutputSize[2], gradOutputSize[3]});
    buffGradInput2 = allocateTensor({inputSize[1]/2, gradOutputSize[2], gradOutputSize[3]});
    
    --print('gradOutputs')
    --print(gradOutput[1])
    --print(gradOutput[2])
    --io.read()
    
    for j = 1, gradOutputSize[2] do
        for k = 1, gradOutputSize[3] do
            for i = 1, inputSize[1]/2 do
                --calculate the gradients with respect to input (f_i & g_i) to the total output
                buffGradInput1[{{i}, {j}, {k}}] = torch.sum(gradOutput1[{{((i-1) * 25) + 1, i * 25}, {j}, {k}}]) + _getGradient(gradOutput2, j, k, i);
                buffGradInput2[{{i}, {j}, {k}}] = torch.sum(gradOutput2[{{((i-1) * 25) + 1, i * 25}, {j}, {k}}]) + _getGradient(gradOutput1, j, k, i);
            end
        end
    end
    
    self.gradInput[{{1, inputSize[1]/2}, {}, {}}] = buffGradInput1;
    self.gradInput[{{(inputSize[1]/2) + 1, inputSize[1]}, {}, {}}] = buffGradInput2;
    --print(self.gradInput:size())    
    return self.gradInput
end 

