import torch
import cv2
import numpy as np

from torchvision import datasets, transforms, models
import torch.nn.functional as F
import torch.nn as nn
import torch.onnx

import onnxruntime as onnxrt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
map_location =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class BasicBlock(nn.Module):

    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample), ]
        else:
            blocks += [BasicBlock(c_out, c_out), ]
    return nn.Sequential(*blocks)


class Net(nn.Module):
    def __init__(self, num_classes=751, reid=False):
        super(Net, self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(64, 64, 2, False)
        # 32 64 32
        self.layer2 = make_layers(64, 128, 2, True)
        # 64 32 16
        self.layer3 = make_layers(128, 256, 2, True)
        # 128 16 8
        self.layer4 = make_layers(256, 512, 2, True)
        # 256 8 4
        self.avgpool = nn.AvgPool2d((8, 4), 1)
        # 256 1 1
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        # classifier
        x = self.classifier(x)
        return x

# Load the PyTorch model

model_path = 'deep_sort_tensorrt/deep_sort/deep/checkpoint/reid.pt'
state_dict = torch.load(model_path, map_location=map_location)['net_dict']

reid_model = Net(reid=True)
reid_model.load_state_dict(state_dict)
reid_model.to(device)

img = cv2.imread("./test_image.jpg")
img = cv2.resize(img.astype(np.float32)/255., (64, 128))


reid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

img_tensor = reid_transform(img)

reid_model.eval()
with torch.inference_mode():
  pred_torch = reid_model(img_tensor.view(1, 3, 128, 64).to(device))


ONNX_FILE_PATH = './reid.onnx'


#input_shape = (1, 3, 128, 64).to(device)
batch_size = 1
dummy_input = torch.randn(batch_size, 3, 128, 64).to(device)


dynamic_axes= {'input':{0:'batch_size'}, 'output':{0:'batch_size'}} #adding names for better debugging

torch.onnx.export(reid_model, dummy_input, ONNX_FILE_PATH, input_names=['input'],
                  output_names=['output'], export_params=True, dynamic_axes=dynamic_axes)

ONNX_FILE_PATH = './reid.onnx'
onnx_session = onnxrt.InferenceSession(ONNX_FILE_PATH,
                                   providers=onnxrt.get_available_providers())

def to_numpy(tensor):
  return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

input_name = onnx_session.get_inputs()[0].name
print("input name", input_name)
input_shape = onnx_session.get_inputs()[0].shape
print("input shape", input_shape)
input_type = onnx_session.get_inputs()[0].type
print("input type", input_type)

output_name = onnx_session.get_outputs()[0].name
print("output name", output_name)
output_shape = onnx_session.get_outputs()[0].shape
print("output shape", output_shape)
output_type = onnx_session.get_outputs()[0].type
print("output type", output_type)

def reid_preprocess_image(img_path):
  
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

  img = cv2.imread(img_path)
  img = cv2.resize(img.astype(np.float32)/255., (64, 128))
  img_tensor = transform(img)
  img_batch = torch.unsqueeze(img_tensor, 0)
  
  return img_batch

img_batch = reid_preprocess_image("./test_image.jpg")

# compute ONNX Runtime output prediction
ort_inputs = {onnx_session.get_inputs()[0].name: to_numpy(img_batch)}
ort_outs = onnx_session.run(None, ort_inputs)


# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(pred_torch), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
