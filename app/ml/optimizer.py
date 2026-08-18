import torch
import torch.optim as optim
from typing import Callable

def get_optimizer(image: torch.Tensor) -> torch.optim.Optimizer:
    return optim.LBFGS([image], max_iter=1)

def get_closure(optimizer, image, feature_extractor, content_features, style_grams, style_weights, alpha, beta, tv_weight) -> Callable:
    from app.ml.losses import calc_content_loss, calc_style_loss, calc_total_variation_loss
    
    def closure():
        optimizer.zero_grad()
        
        target_features = feature_extractor(image)
        
        content_loss = calc_content_loss(content_features['conv4_2'], target_features['conv4_2'])
        style_loss = calc_style_loss(style_grams, target_features, style_weights)
        tv_loss = calc_total_variation_loss(image)
        
        total_loss = (alpha * content_loss) + (beta * style_loss) + (tv_weight * tv_loss)
        total_loss.backward()
        
        return total_loss
        
    return closure
