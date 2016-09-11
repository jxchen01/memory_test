require 'nn'
require 'rnn'
require 'nngraph'
require 'optim'
require 'cutorch'
require 'cunn'
require 'dp'
--require 'cudnn'

--cudnn.benchmark = true
--cudnn.fastest = true

torch.setnumthreads(1) -- speed up
torch.setdefaulttensortype('torch.FloatTensor')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('--rho',5,'maximum length of the sequence for each training iteration')
cmd:option('--learningRate',0.001,'initial learning rate')
cmd:option('--minLR',0.00001,'minimal learning rate')
cmd:option('--gpu',3,'gpu device to use')
cmd:option('--clip',5,'max allowed gradient norm in BPTT')
cmd:option('--nIteration',30,'the number of training iterations')
cmd:option('--s',false,'use sequencer of not')
cmd:text()
opt = cmd:parse(arg or {})

local imageSize = 100

-- set GPU device
cutorch.setDevice(opt.gpu)

-------------------------------------------------------------------------------
---  Part1: Data Loading 
-------------------------------------------------------------------------------
image_all = {}  -- "data" should be a table of data structures with field ('input','target','init')  
target_all = {}
-- load all data
for i=1,10 do
    table.insert(image_all, torch.Tensor(20,imageSize, imageSize):fill(1))
	table.insert(target_all,torch.ByteTensor(20,imageSize,imageSize):fill(1))
end

print('finish loading data')

-------------------------------------------------------------------------------
---  Part2: Model and Criterion
-------------------------------------------------------------------------------
-- build the model 
input = nn.Identity()()

L1a=nn.SpatialConvolution(1,64,3,3,1,1,1,1)(input)
L1b=nn.ReLU(true)(L1a)

L1c=nn.SpatialConvolution(64,256,3,3,1,1,1,1)(L1b)
L1d=nn.ReLU(true)(L1c)

L1e=nn.SpatialConvolution(256,1024,3,3,1,1,1,1)(L1d)
L1f=nn.ReLU(true)(L1e)

L1g=nn.SpatialConvolution(1024,4,3,3,1,1,1,1)(L1f)
L1h=nn.ReLU(true)(L1g)

L1i=nn.Transpose({1,2},{2,3})(L1h)
L1j=nn.Reshape(imageSize*imageSize,4)(L1i)

model = nn.gModule({input},{L1j})
if opt.s then
	model = nn.Sequencer(model)
else
	model = nn.Recursor(model,opt.rho)
end

model:cuda()

if opt.s then
	criterion = nn.SequencerCriterion(nn.CrossEntropyCriterion()):cuda()
else 
	criterion = nn.CrossEntropyCriterion():cuda()
end


-- parameters initialization
params, gradParams = model:getParameters()
params:uniform(-0.01,0.01)


-------------------------------------------------------------------------------
---  Part3: Training 
-------------------------------------------------------------------------------
local optim_config = {learningRate = opt.learningRate, alpha=0.9, epsilon=0.00001}
epoch = 1

function train()

	if epoch>1 then
		if optim_config.learningRate > opt.minLR then
			optim_config.learningRate = optim_config.learningRate * 0.5
		end
	end

    local data_index = torch.randperm(10):long() -- feed the training sequences in a random order
	
	for i=1,2 do
		local input_sequence = image_all[data_index[i]]
		local target_sequence = target_all[data_index[i]]
		
		model:forget()

		local freeMemory, totalMemory = cutorch.getMemoryUsage(opt.gpu)
		print('check 0')
		print(freeMemory)
		print(totalMemory)

		for j=opt.rho,input_sequence:size(1) do
			
			local feval = function (x)
    			if x ~= params then params:copy(x) end
    			gradParams:zero()

				local freeMemory, totalMemory = cutorch.getMemoryUsage(opt.gpu)
				print('check 1')
				print(freeMemory)

				local input_list, target_list = {}, {}
				for k=j-opt.rho+1,j do
					table.insert(input_list, torch.reshape(input_sequence[k],1,1,imageSize,imageSize):cuda())
					table.insert(target_list, torch.reshape(target_sequence[k],1,imageSize*imageSize):cuda())
				end

				local outputs, err = {}, 0
				local gradOutputs, gradInputs = {}, {}

				if opt.s then
					outputs=model:forward(input_list)
				    err = criterion:forward(outputs,target_list)
					gradOutputs = criterion:backward(outputs,target_list)
					gradInputs = model:backward(input_list, gradOutputs)

					local freeMemory, totalMemory = cutorch.getMemoryUsage(opt.gpu)
					print('check 2')
					print(freeMemory)
				else

   					for step=1,opt.rho do
      					outputs[step] = model:forward(input_list[step])
      					err = err + criterion:forward(outputs[step], target_list[step])
   					end

				
   					for step=opt.rho,1,-1 do -- reverse order of forward calls
      					gradOutputs[step] = criterion:backward(outputs[step], target_list[step])
      					gradInputs[step] = model:backward(input_list[step], gradOutputs[step])
						local freeMemory, totalMemory = cutorch.getMemoryUsage(opt.gpu)
						print('check 2')
						print(freeMemory)
   					end
				end

				if opt.clip>0 then
					gradParams:clamp(-opt.clip, opt.clip)
				end
				return err, gradParams
			end

			local _, loss = optim.rmsprop(feval, params, optim_config)
			print('Epoch '..epoch..', Seq:'..data_index[i]..', Frame:'..j..', Loss = '..loss[1])
			-- clean 
    		collectgarbage()			
		end
	end

	epoch = epoch + 1

    -- clean 
    collectgarbage()
end

while epoch < opt.nIteration do
   train()
end


