import litserve as ls
from transformers import AutoModelForImageSegmentation
import torch
from skimage import io
from utils import preprocess_image, postprocess_image
import numpy as np
from fastapi.responses import Response, JSONResponse
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import os
import uuid
import base64
from pydantic import BaseModel

# Create directories if they don't exist
os.makedirs("static/images", exist_ok=True)

# Create Pydantic model for request
class ImageRequest(BaseModel):
    image_url: str

class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        self.model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
        # Always use CPU
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def decode_request(self, request):
        return request["input"]

    def predict(self, image_path: str):
        self.orig_im = io.imread(image_path)
        self.orig_im_size = self.orig_im.shape[0:2]
        model_input_size = [1024, 1024]
        image = preprocess_image(self.orig_im, model_input_size).to(self.device)
        self.result = self.model(image)

    def encode_response(self, output):
        result_image = postprocess_image(self.result[0][0], self.orig_im_size)
        
        # Ensure the mask is binary (0 or 1)
        mask = (result_image > 0.5).astype(np.uint8)
        
        # Create RGBA image with transparency
        rgba = np.zeros((self.orig_im.shape[0], self.orig_im.shape[1], 4), dtype=np.uint8)
        # Copy RGB channels from original image
        rgba[:,:,:3] = self.orig_im
        # Use mask as alpha channel (255 for foreground, 0 for background)
        rgba[:,:,3] = mask * 255
        
        # Convert numpy array to PIL Image with transparency
        pil_image = Image.fromarray(rgba)
        
        # Save the image to a BytesIO object as PNG (to preserve transparency)
        img_byte_arr = BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Return a Response object with the transparent image
        return Response(content=img_byte_arr, media_type="image/png")

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the model
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
    device = torch.device("cpu")
    model.to(device)

@app.post("/api/remove-background")
async def remove_background(request: ImageRequest):
    try:
        # Read the image
        orig_im = io.imread(request.image_url)
        orig_im_size = orig_im.shape[0:2]
        
        # Preprocess
        model_input_size = [1024, 1024]
        image = preprocess_image(orig_im, model_input_size).to("cpu")
        
        # Run model
        result = model(image)
        
        # Postprocess
        result_image = postprocess_image(result[0][0], orig_im_size)
        mask = (result_image > 0.5).astype(np.uint8)
        
        # Create RGBA image with transparency
        rgba = np.zeros((orig_im.shape[0], orig_im.shape[1], 4), dtype=np.uint8)
        rgba[:,:,:3] = orig_im
        rgba[:,:,3] = mask * 255
        
        # Generate unique filename
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join("static/images", filename)
        
        # Save image
        Image.fromarray(rgba).save(filepath)
        
        # Create image URL
        image_url = f"/static/images/{filename}"
        
        # Return JSON response
        return JSONResponse(
            content={
                "status_code": 200,
                "image_url": image_url
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Read file contents
        contents = await file.read()
        
        # Save the original image temporarily
        temp_filename = f"temp_{uuid.uuid4()}.png"
        temp_filepath = os.path.join("static/images", temp_filename)
        with open(temp_filepath, "wb") as f:
            f.write(contents)
        
        # Read the image with skimage for processing
        orig_im = io.imread(temp_filepath)
        orig_im_size = orig_im.shape[0:2]
        
        # Preprocess
        model_input_size = [1024, 1024]
        image = preprocess_image(orig_im, model_input_size).to("cpu")
        
        # Run model
        result = model(image)
        
        # Postprocess
        result_image = postprocess_image(result[0][0], orig_im_size)
        mask = (result_image > 0.5).astype(np.uint8)
        
        # Create RGBA image with transparency
        rgba = np.zeros((orig_im.shape[0], orig_im.shape[1], 4), dtype=np.uint8)
        rgba[:,:,:3] = orig_im
        rgba[:,:,3] = mask * 255
        
        # Convert to PIL image and then to base64
        pil_image = Image.fromarray(rgba)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Remove temporary file
        try:
            os.remove(temp_filepath)
        except:
            pass
        
        # Return JSON response with only base64 image
        return JSONResponse(
            content={
                "status_code": 200,
                # "image_base64": f"data:image/png;base64,{img_str}"
                "image_base64": img_str
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Server setup with updated format according to warning
if __name__ == "__main__":
    api = SimpleLitAPI(max_batch_size=1)
    server = ls.LitServer(api, accelerator="cpu")
    
    # Run both servers (LitServer and FastAPI)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)