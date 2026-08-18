import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from typing import List, Dict, Optional

class VGGFeatureExtractor(nn.Module):
    """
    VGG-19 based feature extractor for Neural Style Transfer.
    """
    def __init__(self, content_layers: Optional[List[str]] = None, style_layers: Optional[List[str]] = None):
        super().__init__()
        
        self.content_layers = content_layers or ['conv4_2']
        self.style_layers = style_layers or ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

        # Load pretrained VGG19 and freeze weights
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        for param in vgg.parameters():
            param.requires_grad = False
            
        self.model = nn.Sequential()
        i = 1
        j = 1
        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):
                name = f'conv{i}_{j}'
                j += 1
            elif isinstance(layer, nn.ReLU):
                name = f'relu{i}_{j-1}'
                # Replace inplace ReLU to prevent memory/gradient issues
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool{i}'
                i += 1
                j = 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn{i}_{j}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

            self.model.add_module(name, layer)

            all_required = self.content_layers + self.style_layers
            if name == sorted(all_required)[-1]: 
                break

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        for name, layer in self.model.named_children():
            x = layer(x)
            if name in self.content_layers or name in self.style_layers:
                features[name] = x
        return features
