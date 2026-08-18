from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import io
import traceback

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
    alpha: float = Form(1.0),
    beta: float = Form(1000000.0),
    preserve_colors: bool = Form(False)
):
    try:
        content_bytes = await content_image.read()
        style_bytes = await style_image.read()
        
        # Run optimization
        result_pil = pipeline.run(
            content_bytes=content_bytes,
            style_bytes=style_bytes,
            alpha=alpha,
            beta=beta,
            preserve_colors=preserve_colors,
            num_steps=250  # Increased for much better quality
        )
        
        img_byte_arr = io.BytesIO()
        result_pil.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/jpeg")
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
