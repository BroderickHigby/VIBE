import io
import numpy
from torch import nn
import torch.onnx
from lib.models.vibe import VIBE_Demo

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch_model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=False,
    ).to(device)

model = 'data/vibe_data/vibe_model_w_3dpw.pth.tar'
batch_size = 1
map_location = lambda storage, loc: storage 
if torch.cuda.is_available():
    map_location = None
torch_model.load_state_dict(torch.load(model, map_location=torch.device('cpu')), strict=False)
torch_model.eval()

x = torch.randn(batch_size, 16, 3, 2048, 2048)
torch_out = torch_model(x)
torch.onnx.export(torch_model, 
					x,
					"vibe.onnx", 
					export_params=True,
					opset_version=10,
					do_constant_folding=True, # constant folding for optimization
					input_names = ['input'],
					output_names = ['output'],
					dynamic_axes={'input' : {0 : 'batch_size'},
									'output' : {0 : 'batch_size'}})

