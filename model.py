import torch.nn as nn
import torchvision

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
import time



class ResNet(nn.Module):
  def __init__(self, model='resnet18', n_channels=4, n_filters=64, n_classes=2, kernel_size=3, stride=1, padding=1):
    super().__init__()
    self.n_classes = n_classes
    models_dict = {
      'resnet8': ResNet8,
      'resnet18': torchvision.models.resnet18,
      'resnet34': torchvision.models.resnet34,
      'resnet50': torchvision.models.resnet50,
      'resnet101': torchvision.models.resnet101,
      'resnet152': torchvision.models.resnet152
    }
    self.base_model = models_dict[model](weights=None)
    self._feature_vector_dimension = self.base_model.fc.in_features
    self.base_model.conv1 = nn.Conv2d(n_channels, n_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    self.base_model = nn.Sequential(*list(self.base_model.children())[:-1]) # Remove the final fully connected layer
    self.fc = nn.Linear(self._feature_vector_dimension, n_classes)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.base_model(x)
    features = x.view(x.size(0), -1)
    x = self.fc(features)
    return x, features

# runs model
def run_model(model, device, images, batch_size_inference=512):

    return generate_predictions(model,device,images,batch_size_inference)

from torch.nn.functional import softmax

def generate_predictions(model, device, images, batch_size_inference):
    if images.dtype == np.uint8:
        images = images.astype(np.float32)/255.0 # convert to 0-1 if uint8 input

    # build dataset
    dataset = TensorDataset(torch.from_numpy(images), torch.from_numpy(np.ones(images.shape[0])))

    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_inference, shuffle=False)

    # run inference 
    all_predictions = []
    for k, (images, labels) in enumerate(dataloader):
        images = images.float().to(device)
        predictions, _ = model.forward(images)
        predictions = softmax(predictions, dim=1)  # Apply softmax to convert logits to probabilities
        predictions = predictions.detach().cpu().numpy()
        all_predictions.append(predictions)
        del images
        del labels
        #torch.cuda.empty_cache()  # Clear GPU memory
        
    del dataset
    del dataloader
    predictions = np.vstack(all_predictions)
  

    return predictions


# ----------------------- for resnet 8 architecture -----------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
class ResNet8(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet8, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 1)
        self.layer2 = self._make_layer(BasicBlock, 128, 1, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# ----------------------- for resnet 8 architecture -----------------------


# ----------------------- v8_hardneg_single (Heguang) -----------------------
# Mirrors malaria-detection/models/v8_hardneg_single/src/model.py exactly so
# the checkpoint loads with matching state_dict keys (backbone.*, fc.*).
import torch.nn.functional as F

class ResNetV8(nn.Module):
    def __init__(self, n_channels=4, n_classes=2):
        super().__init__()
        base = torchvision.models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        feat_dim = base.fc.in_features
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        h = self.backbone(x)
        return self.fc(h.flatten(1))


def _per_image_normalize(x):
    """Per-patch per-channel standardize over (H, W). x is (B, C, H, W) float tensor."""
    m = x.mean(dim=(2, 3), keepdim=True)
    s = x.std(dim=(2, 3), keepdim=True).clamp(min=1e-4)
    return (x - m) / s


def load_v8_model(checkpoint_path, device):
    """Load v8_hardneg_single from a .pt that may be a raw state_dict or a wrapper dict."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        state = ckpt['model_state']
    else:
        state = ckpt
    model = ResNetV8(n_channels=4, n_classes=2).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def run_model_v8(model, device, images, batch_size_inference=512):
    """Score patches with the v8 model. Returns (N,) prob_pos (class-1 softmax).
       Mirrors malaria-detection/models/v8_hardneg_single/src/predict.py."""
    if images.dtype != np.uint8:
        if images.max() <= 1.5:
            images_u8 = (np.clip(images, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            images_u8 = np.clip(images, 0, 255).astype(np.uint8)
    else:
        images_u8 = images

    n = len(images_u8)
    probs = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size_inference):
            batch = np.ascontiguousarray(images_u8[i:i + batch_size_inference]).copy()
            x = torch.from_numpy(batch).to(device).float() / 255.0
            x = _per_image_normalize(x)
            logits = model(x)
            probs[i:i + batch_size_inference] = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs

