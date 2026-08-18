import torch
import torch.nn.functional as F

def calc_content_loss(content_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(target_features, content_features)

def calc_gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    gram = torch.mm(features, features.t())
    return gram.div(b * c * h * w)

def calc_style_loss(style_grams: dict, target_features: dict, style_weights: dict) -> torch.Tensor:
    style_loss = 0
    for layer in style_grams:
        target_feature = target_features[layer]
        target_gram = calc_gram_matrix(target_feature)
        style_gram = style_grams[layer]
        
        layer_loss = F.mse_loss(target_gram, style_gram)
        style_loss += style_weights.get(layer, 1.0) * layer_loss
        
    return style_loss

def calc_total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    tv_h = torch.sum(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    tv_w = torch.sum(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    return tv_h + tv_w

def match_colors(content_img: torch.Tensor, style_img: torch.Tensor) -> torch.Tensor:
    # Placeholder for color matching experiment
    return style_img
