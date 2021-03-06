require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'optim'
local nninit = require 'nninit'

require 'adaptive_ResidualDrop'

-- Saves 40% time according to http://torch.ch/blog/2016/02/04/resnets.html
cudnn.fastest = true
cudnn.benchmark = true

opt = lapp[[
  --maxEpochs     (default 500)         Maximum number of epochs to train the network
  --batchSize     (default 128)         Mini-batch size
  --N             (default 18)          Model has 6*N+2 convolutional layers
  --dataset       (default cifar10)     Use cifar10, cifar100 or svhn
  --deathMode     (default lin_decay)   Use lin_decay or uniform
  --deathRate     (default 0)           1-p_L for lin_decay, 1-p_l for uniform, 0 is constant depth
  --device        (default 0)           Which GPU to run on, 0-based indexing
  --augmentation  (default true)        Standard data augmentation (CIFAR only), true or false
  --trainAlphas   (default true)        Whether to train the alphas at all
  --alphaLR       (default 1)           Learning rate for alphas
  --warmStartEpochs (default 0)         Number of epochs to wait before optimizing alphas
  --devOnTrain    (default false)       Whether to use train set for dev set. If not, use valid set.
  --stochastic    (default true)        Whether to use stochastic layers or pure residuals.
  --resultFolder  (default "")          Path to the folder where you'd like to save results
  --dataRoot      (default "")          Path to data (e.g. contains cifar10-train.t7)
  --trsize        (default 45000)       Size of training data set
  --vasize        (default 5000)        Size of validation data set
  --trainPerDev   (default 1)           Number of training steps per dev step
  --baseLR        (default 0.1)         Base learning rate for parameters
]]
print(opt)
cutorch.setDevice(opt.device+1)   -- torch uses 1-based indexing for GPU, so +1
cutorch.manualSeed(1)
torch.manualSeed(1)
torch.setnumthreads(1)            -- number of OpenMP threads, 1 is enough

opt.resultFolder = string.format('results_tr%d_va%d', opt.trsize, opt.vasize)
os.execute('mkdir -p ' .. opt.resultFolder)
f_err = assert(io.open(string.format('%s/err.out', opt.resultFolder), 'w'))
f_drop = assert(io.open(string.format('%s/drop.out', opt.resultFolder), 'w'))
f_alpha = assert(io.open(string.format('%s/alpha.out', opt.resultFolder), 'w'))

---- Loading data ----
if opt.dataset == 'svhn' then require 'svhn-dataset' else require 'cifar-dataset' end
all_data, all_labels = get_Data(opt.dataset, opt.dataRoot, true)  -- default do shuffling
dataTrain = Dataset.LOADER(all_data, all_labels, "train", opt)
dataValid = Dataset.LOADER(all_data, all_labels, "valid", opt)
dataTest = Dataset.LOADER(all_data, all_labels, "test", opt)
local mean,std = dataTrain:preprocess()
dataValid:preprocess(mean,std)
dataTest:preprocess(mean,std)
print("Training set size:\t",   dataTrain:size())
print("Validation set size:\t", dataValid:size())
print("Test set size:\t\t",     dataTest:size())

---- Optimization hyperparameters ----
sgdState = {
   weightDecay   = 1e-4,
   momentum      = 0.9,
   dampening     = 0,
   nesterov      = true,
}

dev_sgdState = {
   weightDecay   = 0.0,
   momentum      = 0.0,
   dampening     = 0,
   nesterov      = false,
}

-- Point at which learning rate decrease by 10x
lrSchedule = {svhn     = {0.6, 0.7 },
              cifar10  = {0.5, 0.75},
              cifar100 = {0.5, 0.75}}

---- Buidling the residual network model ----
-- Input: 3x32x32
print('Building model...')
model = nn.Sequential()
model.num_blocks = 0
------> 3, 32,32
model:add(cudnn.SpatialConvolution(3, 16, 3,3, 1,1, 1,1)
            :init('weight', nninit.kaiming, {gain = 'relu'})
            :init('bias', nninit.constant, 0))
model:add(cudnn.SpatialBatchNormalization(16))
model:add(cudnn.ReLU(true))
------> 16, 32,32   First Group
for i=1,opt.N do   addResidualDrop(model, opt, 16)   end
------> 32, 16,16   Second Group
addResidualDrop(model, opt, 16, 32, 2)
for i=1,opt.N-1 do   addResidualDrop(model, opt, 32)   end
------> 64, 8,8     Third Group
addResidualDrop(model, opt, 32, 64, 2)
for i=1,opt.N-1 do   addResidualDrop(model, opt, 64)   end
------> 10, 8,8     Pooling, Linear, Softmax
model:add(nn.SpatialAveragePooling(8,8)):add(nn.Reshape(64))
if opt.dataset == 'cifar10' or opt.dataset == 'svhn' then
  model:add(nn.Linear(64, 10))
elseif opt.dataset == 'cifar100' then
  model:add(nn.Linear(64, 100))
else
  print('Invalid argument for dataset!')
end
model:add(cudnn.LogSoftMax())
model:cuda()

loss = nn.ClassNLLCriterion()
loss:cuda()
collectgarbage()
-- print(model)   -- if you need to see the architecture, it's going to be long!

---- Determines the position of all the residual blocks ----
addtables = {}
for i=1,model:size() do
    if tostring(model:get(i)) == 'nn.ResidualDrop' then addtables[#addtables+1] = i end
end

---- Sets the deathRate (1 - survival probability) for all residual blocks  ----

---- Resets all gates to open ----
function openAllGates()
  for i,block in ipairs(addtables) do model:get(block).gate = true end
end

function disable_stochastic()
  for i,block in ipairs(addtables) do model:get(block).no_stochastic = true end
end

function dev()
  for i,block in ipairs(addtables) do model:get(block).dev = true end
end

function undev()
  for i,block in ipairs(addtables) do model:get(block).dev = false end
end

function getAlphas()
  alphas = {}
  for i,block in ipairs(addtables) do
    alpha = model:get(block).alpha_learner:get(1).bias[1]
    alphas[#alphas + 1] = alpha
  end
  return alphas
end

function getBiases()
  biases = {}
  for i,block in ipairs(addtables) do
    bias = model:get(block).net:get(1).bias[1]
    biases[#biases + 1] = bias
  end
  return biases
end

function getDropProbs()
  probs = {}
  for i,block in ipairs(addtables) do
    prob = model:get(block).alpha_learner:forward(model:get(block).zero)[1]
    probs[#probs + 1] = prob
  end
  return probs
end

function getAlphaGradients()
  alphas = {}
  for i,block in ipairs(addtables) do
    alpha = model:get(block).alpha_learner:get(1).gradBias[1]
    alphas[#alphas + 1] = alpha
  end
  return alphas
end

function printAlphas()
  for i, v in ipairs(getAlphas()) do
    io.write(v .. ' ')
    f_alpha:write(v .. ' ')
    f_alpha:flush()
  end
  io.write('\n')
  f_alpha:write('\n')
  f_alpha:flush()
end

function printAlphaGradients()
  for i, v in ipairs(getAlphaGradients()) do
    io.write(v .. ' ')
    f_alpha:write(v .. ' ')
    f_alpha:flush()
  end
  io.write('\n')
  f_alpha:write('\n')
  f_alpha:flush()
end

function printBiases()
  for i, v in ipairs(getBiases()) do
    io.write(v .. ' ')
    f_alpha:write(v .. ' ')
    f_alpha:flush()
  end
  io.write('\n')
  f_alpha:write('\n')
  f_alpha:flush()
end

function printDropProbs()
  for i, v in ipairs(getDropProbs()) do
    io.write(v .. ' ')
    f_drop:write(v .. ' ')
    f_drop:flush()
  end
  io.write('\n')
  f_drop:write('\n')
  f_drop:flush()
end

if opt.no_stochastic then
  disable_stochastic()
end

---- Testing ----
function evalModel(dataset)
  model:evaluate()
  openAllGates() -- this is actually redundant, test mode never skips any layer
  local correct = 0
  local total = 0
  local batches = torch.range(1, dataset:size()):long():split(opt.batchSize)
  for i=1,#batches do
     local batch = dataset:sampleIndices(batches[i])
     local inputs, labels = batch.inputs, batch.outputs:long()
     local y = model:forward(inputs:cuda()):float()
     local _, indices = torch.sort(y, 2, true)
     -- indices is a tensor with shape (batchSize, nClasses)
     local top1 = indices:select(2, 1)
     correct = correct + torch.eq(top1, labels):sum()
     total = total + indices:size(1)
  end
  return 1-correct/total
end

-- Saving and printing results
all_results = {}  -- contains test and validation error throughout training
-- For CIFAR, accounting is done every epoch, and for SVHN, every 200 iterations
function accounting(training_time, train_accuracy)
  local results = {train_accuracy, evalModel(dataValid), evalModel(dataTest)}
  all_results[#all_results + 1] = results
  -- Saves the errors. These get covered up by new ones every time the function is called
  torch.save(opt.resultFolder .. '/' .. string.format('errors_%d_%s_%s_%.1f',
    opt.N, opt.dataset, opt.deathMode, opt.deathRate), all_results)
  if opt.dataset == 'svhn' then
    out = string.format('Iter %d:\t%.2f%%\t\t%.2f%%\t\t%.2f%%\t\t%0.0fs',
      sgdState.iterCounter, results[1]*100, results[2]*100, results[3]*100, training_time)
    print(out)
    f_err:write(out .. '\n')
    f_err:flush()
  else
    out = string.format('Epoch %d:\t%.2f%%\t\t%.2f%%\t\t%.2f%%\t\t%0.0fs',
      sgdState.epochCounter, results[1]*100, results[2]*100, results[3]*100, training_time)
    print(out)
    f_err:write(out .. '\n')
    f_err:flush()
  end
  -- printDropProbs()
  -- printAlphaGradients()
  -- print('Printing biases')
  -- printBiases()
end

-- TODO: add a function to do a forward pass on the validation set and backprop w.r.t. the alphas

---- Training ----
function main()

  opt.maxEpochs = math.floor(opt.maxEpochs * (45000 / opt.trsize))

  undev()
  local weights, gradients = model:getParameters()
  dev()
  local alpha_weights, alpha_gradients = model:getParameters()

  sgdState.epochCounter  = 1
  if opt.dataset == 'svhn' then
    sgdState.iterCounter = 1
    print('Training...\nIter\t\tTrain. err\tValid. err\tTest err\tTraining time')
  else
    print('Training...\nEpoch\tTrain.err\tValid. err\tTest err\tTraining time')
  end
  local all_indices = torch.range(1, dataTrain:size())
  local valid_indices = torch.range(1, dataValid:size())
  local timer = torch.Timer()
  if opt.trainAlphas then
    additional_epochs = 100
  else
    additional_epochs = 0
  end

  while sgdState.epochCounter <= opt.maxEpochs + additional_epochs do
    -- Learning rate schedule
    if sgdState.epochCounter < opt.maxEpochs*lrSchedule[opt.dataset][1] then
      sgdState.learningRate = opt.baseLR
      dev_sgdState.learningRate = opt.alphaLR
    elseif sgdState.epochCounter < opt.maxEpochs*lrSchedule[opt.dataset][2] then
      sgdState.learningRate = 0.1 * opt.baseLR
      dev_sgdState.learningRate = opt.alphaLR * 0.1
    elseif sgdState.epochCounter < opt.maxEpochs then
      sgdState.learningRate = 0.01 * opt.baseLR
      dev_sgdState.learningRate = opt.alphaLR * 0.01
    else
      sgdState.learningRate = 0.01 * opt.baseLR * (torch.sqrt((sgdState.epochCounter - opt.maxEpochs)/100))
      dev_sgdState.learningRate = 0
    end

    local shuffle = torch.randperm(dataTrain:size())
    local batches = all_indices:index(1, shuffle:long()):long():split(opt.batchSize)
    local valid_shuffle = torch.randperm(dataValid:size())
    local valid_batches = valid_indices:index(1, valid_shuffle:long()):long():split(opt.batchSize)

    correct = 0
    total = 0

    for i=1,#batches do
        model:training()
        openAllGates()    -- resets all gates to open
        -- Randomly determines the gates to close, according to their survival probabilities
        for i,tb in ipairs(addtables) do
          if torch.rand(1):cuda()[1] > model:get(tb).alpha_learner:forward(model:get(tb).zero)[1] then
            model:get(tb).gate = false
          end
        end

        undev()
        gradients:zero()

        function train_eval(x)
            gradients:zero()
            local batch = dataTrain:sampleIndices(batches[i])
            local inputs, long_labels = batch.inputs, batch.outputs:long()
            inputs = inputs:cuda()
            labels = long_labels:cuda()
            local y = model:forward(inputs)
            local loss_val = loss:forward(y, labels)
            local dl_df = loss:backward(y, labels)
            model:backward(inputs, dl_df)
            local _, indices = torch.sort(y:float(), 2, true)
            -- indices is a tensor with shape (batchSize, nClasses)
            local top1 = indices:select(2, 1)
            correct = correct + torch.eq(top1, long_labels):sum()
            total = total + indices:size(1)

            -- print(torch.sum(gradients))

            return loss_val, gradients
        end

        -- first do a step of sgd against the net parameters. by turning off dev we fix alphas..
        optim.sgd(train_eval, weights, sgdState)

        train_accuracy = 1 - correct/total

        -- now open gates, set dev mode
        openAllGates()

        if (i % opt.trainPerDev == 0) and opt.trainAlphas and sgdState.epochCounter >= opt.warmStartEpochs then
          dev()

          -- i have no idea if we really need to redefine this fn or could use train_eval from before.
          -- will need to test it out.
          function dev_eval(x)
              alpha_gradients:zero()
              -- get the i'th valid_batch, modulo # valid_batches
              -- (as # train_batches exceeds #valid_batches)
              local batch = dataValid:sampleIndices(valid_batches[1 + (i % #valid_batches) ])
              if opt.devOnTrain then
                batch = dataTrain:sampleIndices(batches[i])
              end
              local inputs, labels = batch.inputs, batch.outputs:long()
              inputs = inputs:cuda()
              labels = labels:cuda()
              local y = model:forward(inputs)
              local loss_val = loss:forward(y, labels)
              local dl_df = loss:backward(y, labels)
              model:backward(inputs, dl_df)
              return loss_val, alpha_gradients
          end

          -- now do step of sgd against alphas. by setting dev mode we fix net parameters.
          optim.sgd(dev_eval, alpha_weights, dev_sgdState)
        end

        if opt.dataset == 'svhn' then
          if sgdState.iterCounter % 200 == 0 then
            accounting(timer:time().real, train_accuracy)
            timer:reset()
          end
          sgdState.iterCounter = sgdState.iterCounter + 1
        end
    end
    if opt.dataset ~= 'svhn' then
      accounting(timer:time().real, train_accuracy)
      timer:reset()
    end

    -- periodic model saving
    if sgdState.epochCounter % 1000 == 0 then
      torch.save(opt.resultFolder .. '/' .. string.format('model_%d_%s_%s_%.1f_%d', opt.N, opt.dataset, opt.deathMode, opt.deathRate, sgdState.epochCounter), model)
    end

    sgdState.epochCounter = sgdState.epochCounter + 1
  end

  -- Saves the the last model, optional. Model loading feature is not available now but is easy to add
   torch.save(opt.resultFolder .. '/' .. string.format('last_model_%d_%s_%s_%.1f_%d', opt.N, opt.dataset, opt.deathMode, opt.deathRate, sgdState.epochCounter-1), model)
end

main()
