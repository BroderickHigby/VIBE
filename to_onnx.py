import torch
import torch.onnx



# ========= Load pretrained weights ========= #
#pretrained_file = download_ckpt(use_3dpw=False)
#ckpt = torch.load(pretrained_file) # , map_location=torch.device('cpu'))
#print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
#ckpt = ckpt['gen_state_dict']
# model.load_state_dict(ckpt, strict=False)

# model.eval()
# print(f'Loaded pretrained weights from \"{pretrained_file}\"')

model = torch.load('./data/vibe_data/vibe_model_w_3dpw.pth.tar')
# model.load_state_dict(torch.load('data/vibe_data/vibe_model_w_3dpw.pth.tar'))
with torch.no_grad():
# batch, seqlen, nc, h, w
    x = torch.randn(1, 16, 3, 2048, 2048, device='cuda', requires_grad=True)
    torch_out = model(x)
    # Export the model to onnx 
    print("exporting ONNX model")
    torch.onnx.export(model,               # model being run
                x,                         # model input (or a tuple for multiple inputs)
                "vibe.onnx",               # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=10,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['input'],   # the model's input names
                output_names = ['output'], # the model's output names
                dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})
