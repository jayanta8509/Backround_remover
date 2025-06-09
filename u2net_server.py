from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import base64
from io import BytesIO
from PIL import Image
from backgroundremover.bg import remove
import requests
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/images", exist_ok=True)

MODEL_CHOICES = ["u2net", "u2net_human_seg", "u2netp"]

class ImageRequest(BaseModel):
    image_url: str
    model_name: str = "u2net"

@app.post("/api/upload-image")
async def remove_background(
    file: UploadFile = File(...),
    model_name: str = Form("u2net")
):
    if model_name not in MODEL_CHOICES:
        raise HTTPException(status_code=400, detail=f"Invalid model_name. Choose from {MODEL_CHOICES}")
    try:
        contents = await file.read()
        # Remove background
        result = remove(
            contents,
            model_name=model_name,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_structure_size=10,
            alpha_matting_base_size=1000
        )
        # Save to BytesIO and encode as base64
        img = Image.open(BytesIO(result))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return JSONResponse(content={
            "status_code": 200,
            "image_base64": img_str
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/remove-background")
async def remove_background_url(request: ImageRequest):
    model_name = request.model_name
    if model_name not in MODEL_CHOICES:
        raise HTTPException(status_code=400, detail=f"Invalid model_name. Choose from {MODEL_CHOICES}")
    try:
        # Download the image from the URL
        response = requests.get(request.image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image from URL")
        contents = response.content

        # Remove background
        result = remove(
            contents,
            model_name=model_name,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_structure_size=10,
            alpha_matting_base_size=1000
        )
        # Save to BytesIO and encode as base64
        img = Image.open(BytesIO(result))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return JSONResponse(content={
            "status_code": 200,
            "image_base64": img_str
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 