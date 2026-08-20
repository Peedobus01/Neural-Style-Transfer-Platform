import torch
from app.ml.model import VGGFeatureExtractor
from app.ml.preprocessing import load_image, tensor_to_image
from app.ml.losses import calc_gram_matrix
from app.ml.optimizer import get_optimizer, get_closure

class StyleTransferPipeline:
    def __init__(self, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.feature_extractor = VGGFeatureExtractor().to(self.device).eval()
        
    def run(self, content_bytes, style_bytes, alpha=1e4, beta=1e2, tv_weight=1e-5, preserve_colors=False, num_steps=100, optimizer_type="adam", noise_ratio=0.0):
        content_img = load_image(content_bytes).to(self.device)
        shape = (content_img.shape[2], content_img.shape[3])
        style_img = load_image(style_bytes, shape=shape).to(self.device)
        
        with torch.no_grad():
            content_features = self.feature_extractor(content_img)
            style_features = self.feature_extractor(style_img)
            
        style_grams = {layer: calc_gram_matrix(style_features[layer]) for layer in style_features if layer.startswith('conv') and layer != 'conv4_2'}
        style_weights = {'conv1_1': 0.2, 'conv2_1': 0.2, 'conv3_1': 0.2, 'conv4_1': 0.2, 'conv5_1': 0.2}

        # Scale up the style weight because the normalized Gram Matrix values are extremely small
        effective_beta = beta * 1000.0 

        # Initialize the generated image with optional noise blending
        if noise_ratio > 0.0:
            noise = torch.randn_like(content_img).uniform_(0, 1)
            generated_img = (content_img * (1 - noise_ratio) + noise * noise_ratio).requires_grad_(True).to(self.device)
        else:
            generated_img = content_img.clone().requires_grad_(True).to(self.device)
        
        optimizer = get_optimizer(generated_img, optimizer_type=optimizer_type)
        closure = get_closure(optimizer, generated_img, self.feature_extractor, content_features, style_grams, style_weights, alpha, effective_beta, tv_weight)
        
        if optimizer_type.lower() == "lbfgs":
            run = [0]
            while run[0] <= num_steps:
                def opt_closure():
                    loss = closure()
                    if run[0] % 20 == 0:
                        print(f"Step {run[0]}: Total Loss = {loss.item():.4f}")
                    run[0] += 1
                    return loss
                optimizer.step(opt_closure)
        else:
            # Adam optimization loop (or other standard optimizers)
            for step in range(num_steps):
                loss = closure()
                optimizer.step()
                
                # We must clamp the image to keep Adam stable, but because the image
                # is ImageNet normalized, the valid bounds are not [0, 1]. 
                # We must calculate the valid bounds per channel:
                with torch.no_grad():
                    for c, (mean, std) in enumerate(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
                        generated_img[:, c, :, :].clamp_((0.0 - mean) / std, (1.0 - mean) / std)
                        
                if step % 50 == 0:
                    c_loss = getattr(closure, 'content_loss', 0)
                    s_loss = getattr(closure, 'style_loss', 0)
                    print(f"Epoch {step}: Total={loss.item():.2f} | Content={c_loss:.2f} | Style={s_loss:.2f}")
            
        return tensor_to_image(generated_img)
