from PIL import Image
import torchvision.transforms as transforms
import torch
import io
from typing import Optional, Tuple, Union

# ImageNet normalization parameters required for pretrained VGG
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

def load_image(image_path_or_bytes: Union[str, bytes], max_size: int = 400, shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    if isinstance(image_path_or_bytes, bytes):
        image = Image.open(io.BytesIO(image_path_or_bytes)).convert('RGB')
    else:
        image = Image.open(image_path_or_bytes).convert('RGB')
        
    size = shape if shape is not None else max_size
    if shape is None and max(image.size) > max_size:
        size = max_size
        
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    Reverses the ImageNet normalization and converts the tensor back to a PIL image.
    """
    image = tensor.cpu().clone().detach().squeeze(0)
    image = image * STD + MEAN
    image = image.clamp(0, 1)
    
    transform = transforms.ToPILImage()
    return transform(image)
