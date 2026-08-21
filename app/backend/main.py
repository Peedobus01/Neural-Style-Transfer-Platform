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


@app.post("/api/stylize")
async def stylize_image(
    content_image: UploadFile = File(...),
    style_image: UploadFile = File(...),
    alpha: float = Form(10.0),
    beta: float = Form(100000000.0),
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

from fastapi.responses import FileResponse

@app.get("/")
async def serve_index():
    return FileResponse("app/frontend/index.html")

@app.get("/{filename}")
async def serve_file(filename: str):
    file_path = os.path.join("app/frontend", filename)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "Not found"})

if __name__ == "__main__":
    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=8000, reload=True)
