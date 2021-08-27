import Classification.vgg.arch as arch
from torchviz import make_dot
import torch

model = arch.VGG()

x = torch.randn(1, 3, 224, 224)
y = model(x)
make_dot(y, params=dict(model.named_parameters())).render("rnn_torchviz", format="png")

input_names = ['Sentence']
output_names = ['yhat']
torch.onnx.export(model, (torch.randn(1,3,224,224)), 'rnn.png', input_names=input_names, output_names=output_names)