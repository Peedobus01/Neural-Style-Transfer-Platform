import torch
from app.ml.model import VGGFeatureExtractor
from app.ml.preprocessing import load_image, tensor_to_image
from app.ml.losses import calc_gram_matrix
from app.ml.optimizer import get_optimizer, get_closure

class StyleTransferPipeline:
    def __init__(self, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.feature_extractor = VGGFeatureExtractor().to(self.device).eval()
        
    def run_stream(self, content_bytes, style_bytes, alpha=1e4, beta=1e2, tv_weight=1e-5, preserve_colors=False, num_steps=100, optimizer_type="adam", noise_ratio=0.0, intermediate_frames=0):
        content_img = load_image(content_bytes).to(self.device)
        shape = (content_img.shape[2], content_img.shape[3])
        style_img = load_image(style_bytes, shape=shape).to(self.device)
        
        with torch.no_grad():
            content_features = self.feature_extractor(content_img)
            style_features = self.feature_extractor(style_img)
            
        style_grams = {layer: calc_gram_matrix(style_features[layer]) for layer in style_features if layer.startswith('conv') and layer != 'conv4_2'}
        style_weights = {'conv1_1': 0.2, 'conv2_1': 0.2, 'conv3_1': 0.2, 'conv4_1': 0.2, 'conv5_1': 0.2}

        effective_beta = beta * 1000.0 

        if noise_ratio > 0.0:
            noise = torch.randn_like(content_img).uniform_(0, 1)
            generated_img = (content_img * (1 - noise_ratio) + noise * noise_ratio).requires_grad_(True).to(self.device)
        else:
            generated_img = content_img.clone().requires_grad_(True).to(self.device)
        
        optimizer = get_optimizer(generated_img, optimizer_type=optimizer_type)
        closure = get_closure(optimizer, generated_img, self.feature_extractor, content_features, style_grams, style_weights, alpha, effective_beta, tv_weight)
        
        yield_interval = num_steps if intermediate_frames <= 0 else max(1, num_steps // (intermediate_frames + 1))
        
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
            for step in range(1, num_steps + 1):
                loss = closure()
                optimizer.step()
                
                with torch.no_grad():
                    for c, (mean, std) in enumerate(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
                        generated_img[:, c, :, :].clamp_((0.0 - mean) / std, (1.0 - mean) / std)
                        
                if step % 10 == 0 or step == num_steps:
                    c_loss = getattr(closure, 'content_loss', 0)
                    s_loss = getattr(closure, 'style_loss', 0)
                    print(f"Epoch {step}: Total={loss.item():.2f} | Content={c_loss:.2f} | Style={s_loss:.2f}")
                
                # Yield progress updates and images
                is_image_step = (step % yield_interval == 0) or (step == num_steps)
                is_progress_step = (step % 10 == 0) or is_image_step
                
                if is_progress_step:
                    yield step, num_steps, tensor_to_image(generated_img) if is_image_step else None
            
        if optimizer_type.lower() == "lbfgs":
            yield num_steps, num_steps, tensor_to_image(generated_img)

    def run(self, *args, **kwargs):
        # A wrapper that just returns the final image for backward compatibility
        kwargs['intermediate_frames'] = 0
        final_image = None
        for step, total, img in self.run_stream(*args, **kwargs):
            if img is not None:
                final_image = img
        if final_image is None:
            raise RuntimeError("Optimization failed to produce an image.")
        return final_image
