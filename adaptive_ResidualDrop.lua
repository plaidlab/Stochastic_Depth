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

    -- init_alpha is a CONSTANT and is not trained
    self.init_alpha = init_alpha

    -- alpha_learner facilitates training of alpha
    -- it just adds some bias to init_alpha and returns  sigmoid(init_alpha + bias)
    -- training the bias is equivalent to training alpha

    self.alpha_learner = nn.Sequential()
    self.alpha_learner:add(nn.Add(1))
    self.alpha_learner:add(nn.Sigmoid())

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

   -- Start by adding residual to output
    local skip_forward = self.skip:forward(input)
    self.output:resizeAs(skip_forward):copy(skip_forward)

   -- Output calculation when in dev mode
   -- It's just a weighted version of the normal output
    if self.dev then
      -- note mul must be with a scalar value contained in a tensor, NOT a tensor
      self.output:add(self.net:forward(input):mul(self.alpha_learner:forward(self.init_alpha)[1]))

    -- Output calculation when in train mode
    -- Add net:forward if gate is open
    elseif self.train then
      if self.gate then -- only compute convolutional output when gate is open
        self.output:add(self.net:forward(input))
      end
    end
    return self.output
end

function ResidualDrop:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput or input.new()
   self.gradInput:resizeAs(input):copy(self.skip:updateGradInput(input, gradOutput))

   -- Gradient calculations of module outputs w.r.t. module inputs when in dev mode
   -- It's just a weighted version of the normal gradient
   if self.dev then
        -- y = x + layer_weighting * f(x) so dy/dx = layer_weighting * df(x)/dx
        -- note mul must be with a scalar value contained in a tensor, NOT a tensor
        self.gradInput:add(self.net:updateGradInput(input, gradOutput):mul(self.alpha_learner.output[1]))

   -- Gradient calculations of module outputs w.r.t. module inputs when in dev mode
   -- It's just the normal gradient calculation, so call net:updateGradInput
   elseif self.train then
        if self.gate then
            -- y = x + f(x) so dy/dx = df(x)/dx
            self.gradInput:add(self.net:updateGradInput(input, gradOutput))
        end
   end

   return self.gradInput
end

function ResidualDrop:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   -- Gradient calculations w.r.t. params if in dev mode (i.e. training alphas)
   -- We avoid training the "net" weights by NOT calling net:accGradParameters
   -- So no gradients are accumulated for net params and sgd will not update them
   if self.dev then
        -- y = x + g(alpha) * f(x) so given dL / dy, dL / dalpha = (dL / dy) (dg/dalpha * f(x)) = (dL / dy * f(x)) * dg/dalpha
        -- note cmul is elementwise
       self.alpha_learner:accGradParameters(self.init_alpha, gradOutput:cmul(self.net.output), scale)

   -- Gradient calculations w.r.t. params in train mode and not in dev mode
   -- We avoid training the "alpha_learner" weights by NOT calling alpha_learner:accGradParameters
   -- So no gradients are accumulated for alpha_learner param and sgd will not update it
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
