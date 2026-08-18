import os
import sys

# Ensure app modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ml.pipeline import StyleTransferPipeline

def run_baseline_experiment():
    print("Running baseline Neural Style Transfer experiment...")
    # Add testing logic here

if __name__ == "__main__":
    run_baseline_experiment()
