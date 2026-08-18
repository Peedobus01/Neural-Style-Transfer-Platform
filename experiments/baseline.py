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
    
    # Point to the storage/uploads folder where you uploaded the files
    content_path = "storage/uploads/content.jpg"
    style_path = "storage/uploads/style.jpg"
    
    import os
    if not os.path.exists(content_path) or not os.path.exists(style_path):
        print(f"ERROR: Could not find images! Make sure they are named EXACTLY 'content.jpg' and 'style.jpg' and are placed inside the 'storage/uploads/' folder.")
        return

    # Read the files as bytes
    with open(content_path, "rb") as f:
        content_bytes = f.read()
    with open(style_path, "rb") as f:
        style_bytes = f.read()

    # Run the pipeline with higher steps for the GPU
    print("Processing image... (this should be very fast on a T4 GPU!)")
    result_img = pipeline.run(
        content_bytes=content_bytes,
        style_bytes=style_bytes,
        alpha=1.0,
        beta=1000000.0,
        num_steps=300
    )
    
    # Save the output
    output_path = "storage/outputs/baseline_result.jpg"
    result_img.save(output_path)
    print(f"Success! Stylized image saved to: {output_path}")
    run_baseline_experiment()
