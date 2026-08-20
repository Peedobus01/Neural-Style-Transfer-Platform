import os
import sys

# Ensure app modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ml.pipeline import StyleTransferPipeline

import torch
from PIL import Image
import io

def run_baseline_experiment():
    print("Starting baseline Neural Style Transfer experiment on GPU...")
    
    # Initialize the pipeline
    pipeline = StyleTransferPipeline()
    
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Run Neural Style Transfer")
    parser.add_argument("--content", default="storage/uploads/content.jpg", help="Path to content image")
    parser.add_argument("--style", default="storage/uploads/style.jpg", help="Path to style image")
    args = parser.parse_args()
    
    content_path = args.content
    style_path = args.style
    
    if not os.path.exists(content_path) or not os.path.exists(style_path):
        print(f"ERROR: Could not find images!\nExpected Content: {content_path}\nExpected Style: {style_path}")
        print("Please rename your images to 'content.jpg' and 'style.jpg' and place them in the storage/uploads/ folder.")
        return
    
    print(f"Found images! \nContent: {content_path}\nStyle: {style_path}")

    # Read the files as bytes
    with open(content_path, "rb") as f:
        content_bytes = f.read()
    with open(style_path, "rb") as f:
        style_bytes = f.read()

    print("Processing image... (Running full 1000 epochs on Adam!)")
    result_img = pipeline.run(
        content_bytes=content_bytes,
        style_bytes=style_bytes,
        alpha=10.0,
        beta=1e8,
        num_steps=2500,
        optimizer_type="adam",
        noise_ratio=0.5
    )
    
    # Save the output
    output_path = "storage/outputs/baseline_result.jpg"
    result_img.save(output_path)
    print(f"Success! Stylized image saved to: {output_path}")

if __name__ == "__main__":
    run_baseline_experiment()
