from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import io
import traceback
import os

from app.ml.pipeline import StyleTransferPipeline

app = FastAPI(title="Neural Style Transfer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = StyleTransferPipeline()

# Ensure storage directories exist
os.makedirs("storage/uploads", exist_ok=True)
os.makedirs("storage/outputs", exist_ok=True)

@app.post("/api/stylize")
async def stylize_image(
    content_image: UploadFile = File(...),
    style_image: UploadFile = File(...),
    alpha: float = Form(10.0),
    beta: float = Form(100000000.0),
    preserve_colors: bool = Form(False),
    num_steps: int = Form(2500),
    intermediate_frames: int = Form(0)
):
    try:
        content_bytes = await content_image.read()
        style_bytes = await style_image.read()
        
        def generate_sse():
            import base64
            import json
            try:
                for step, total, img in pipeline.run_stream(
                    content_bytes=content_bytes,
                    style_bytes=style_bytes,
                    alpha=alpha,
                    beta=beta,
                    preserve_colors=preserve_colors,
                    num_steps=num_steps,
                    intermediate_frames=intermediate_frames
                ):
                    data = {"step": step, "total": total}
                    if img:
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='JPEG', quality=85)
                        b64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                        data["image"] = f"data:image/jpeg;base64,{b64}"
                    
                    yield f"data: {json.dumps(data)}\n\n"
            except Exception as e:
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(generate_sse(), media_type="text/event-stream")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# Mount the static frontend directory so it is served by the backend
app.mount("/", StaticFiles(directory="app/frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=8000, reload=True)
