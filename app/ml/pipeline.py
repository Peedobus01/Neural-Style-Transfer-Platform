import torch
from app.ml.model import VGGFeatureExtractor
from app.ml.preprocessing import load_image, tensor_to_image
from app.ml.losses import calc_gram_matrix
from app.ml.optimizer import get_optimizer, get_closure

class StyleTransferPipeline:
    def __init__(self, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.feature_extractor = VGGFeatureExtractor().to(self.device).eval()
        
    def run(self, content_bytes, style_bytes, alpha=1e4, beta=1e2, tv_weight=1e-5, preserve_colors=False, num_steps=100):
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

        generated_img = content_img.clone().requires_grad_(True).to(self.device)
        
        optimizer = get_optimizer(generated_img)
        closure = get_closure(optimizer, generated_img, self.feature_extractor, content_features, style_grams, style_weights, alpha, effective_beta, tv_weight)
        
        run = [0]
        while run[0] <= num_steps:
            def opt_closure():
                loss = closure()
                if run[0] % 20 == 0:
                    print(f"Step {run[0]}: Total Loss = {loss.item():.4f}")
                run[0] += 1
                return loss
            optimizer.step(opt_closure)
            
        return tensor_to_image(generated_img)
