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
    
    # NOTE: You need to upload two images to your Colab environment or specify their paths here
    content_path = "content.jpg"
    style_path = "style.jpg"
    
    import os
    if not os.path.exists(content_path) or not os.path.exists(style_path):
        print("ERROR: Please upload 'content.jpg' and 'style.jpg' to the folder before running this script.")
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
