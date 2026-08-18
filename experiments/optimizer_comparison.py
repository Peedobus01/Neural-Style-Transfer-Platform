import os
import sys

# Ensure app modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def compare_optimizers():
    print("Comparing Adam vs L-BFGS for Neural Style Transfer...")
    # Add comparison logic here

if __name__ == "__main__":
    compare_optimizers()
