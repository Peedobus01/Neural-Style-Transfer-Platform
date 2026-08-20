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
    
    import glob
    
    # Grab the first two images found in storage/uploads/ regardless of their name
    upload_files = [f for f in glob.glob("storage/uploads/*") if not f.endswith('.keep')]
    
    if len(upload_files) < 2:
        print(f"ERROR: Could not find 2 images! Found {len(upload_files)}. Please put your content and style images in the storage/uploads/ folder.")
        return
        
    content_path = upload_files[0]
    style_path = upload_files[1]
    
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
        alpha=1.0,
        beta=1000000.0,
        num_steps=1000,
        optimizer_type="adam",
        noise_ratio=0.1
    )
    
    # Save the output
    output_path = "storage/outputs/baseline_result.jpg"
    result_img.save(output_path)
    print(f"Success! Stylized image saved to: {output_path}")

if __name__ == "__main__":
    run_baseline_experiment()
