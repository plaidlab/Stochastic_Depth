require 'nn'
require 'cudnn'
require 'cunn'
local nninit = require 'nninit'

local AlphaLearner, parent = torch.class('nn.AlphaLearner', 'nn.Container')


local ResidualDrop, parent = torch.class('nn.ResidualDrop', 'nn.Container')

function ResidualDrop:__init(init_alpha, nChannels, nOutChannels, stride)
    parent.__init(self)
    self.gradInput = torch.Tensor()
    self.gate = true
    self.train = true
    self.dev = false

    -- DEATH RATE HERE
    -- TODO: replace with alpha. ensure this is a trainable variable, but NOT a parameter of the ResidualDrop or net
    self.init_alpha = init_alpha

    self.alpha_learner = nn.Sequential()
    self.alpha_learner.add(nn.Add(1))
    self.alpha_learner.add(nn.Sigmoid())

    -- TODO: replace the computations where deathRate is used with sigmoid(alpha)

    nOutChannels = nOutChannels or nChannels
    stride = stride or 1

    self.net = nn.Sequential()
    self.net:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 3,3, stride,stride, 1,1)
                                             :init('weight', nninit.kaiming, {gain = 'relu'})
                                             :init('bias', nninit.constant, 0))
    self.net:add(cudnn.SpatialBatchNormalization(nOutChannels))
    self.net:add(cudnn.ReLU(true))
    self.net:add(cudnn.SpatialConvolution(nOutChannels, nOutChannels,
                                      3,3, 1,1, 1,1)
                                      :init('weight', nninit.kaiming, {gain = 'relu'})
                                      :init('bias', nninit.constant, 0))
    self.net:add(cudnn.SpatialBatchNormalization(nOutChannels))
    self.skip = nn.Sequential()
    self.skip:add(nn.Identity())
    if stride > 1 then
       -- optional downsampling
       self.skip:add(nn.SpatialAveragePooling(1, 1, stride,stride))
    end
    if nOutChannels > nChannels then
       -- optional padding, this is option A in their paper
       self.skip:add(nn.Padding(1, (nOutChannels - nChannels), 3))
    elseif nOutChannels < nChannels then
       print('Do not do this! nOutChannels < nChannels!')
    end

    self.modules = {self.net, self.skip}
end

function ResidualDrop:updateOutput(input)
    local skip_forward = self.skip:forward(input)
    self.output:resizeAs(skip_forward):copy(skip_forward)
    if self.dev then
      self.output:add(self.net:forward(input):mul(self.alpha_learner.forward(self.init_alpha)))
    elseif self.train then
      if self.gate then -- only compute convolutional output when gate is open
        self.output:add(self.net:forward(input))
      end
    end
    return self.output
end

-- TODO: add gradient calculation for when not self.train
function ResidualDrop:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput or input.new()
   self.gradInput:resizeAs(input):copy(self.skip:updateGradInput(input, gradOutput))
   if self.dev then
        -- y = x + layer_weighting * f(x) so dy/dx = layer_weighting * df(x)/dx
        self.gradInput:add(self.net:updateGradInput(input, gradOutput):mul(self.alpha_learner.output))
   elseif self.train then
        if self.gate then
            -- y = x + f(x) so dy/dx = df(x)/dx
            self.gradInput:add(self.net:updateGradInput(input, gradOutput))
        end
   end
   return self.gradInput
end

-- TODO: add gradient calculation for when not self.train
function ResidualDrop:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if self.dev then
        -- y = x + g(alpha) * f(x) so given dL / dy, dL / dalpha = (dL / dy) (dg/dalpha * f(x))
       self.alpha_learner:accGradParameters(self.init_alpha, gradOutput:mul(self.net.output), scale)
   elseif self.train then
       if self.gate then
          -- y = x + f_theta (x) so given dL / dy, dL / dtheta = (dL / dy) (df(x) / dtheta)
          self.net:accGradParameters(input, gradOutput, scale)
       end
   end
end

---- Adds a residual block to the passed in model ----
function addResidualDrop(model, init_alpha, nChannels, nOutChannels, stride)
   model:add(nn.ResidualDrop(init_alpha, nChannels, nOutChannels, stride))
   model:add(cudnn.ReLU(true))
   return model
end
