# Neural Style Transfer Platform

A robust platform for exploring and running Neural Style Transfer (NST) using PyTorch, FastAPI, and a lightweight Vanilla HTML/JS frontend.

## Architecture

- **`app/frontend/`**: The web interface to upload images and configure parameters.
- **`app/backend/`**: The FastAPI server that handles requests.
- **`app/ml/`**: The core PyTorch engine for feature extraction, losses, and optimization.
- **`experiments/`**: Scripts for testing different optimizers, hyperparameters, and variations (like color preservation).
- **`storage/`**: Local storage for uploaded content/styles and generated outputs.

## Setup

1. Create a virtual environment at the root of the project:
   ```bash
   python -m venv venv
   ```
2. Activate it:
   - Windows: `.\venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
3. Install dependencies:
   ```bash
   pip install -r app/backend/requirements.txt
   ```
4. Run the backend:
   ```bash
   uvicorn app.backend.main:app --reload
   ```
5. Open `app/frontend/index.html` in your web browser.
