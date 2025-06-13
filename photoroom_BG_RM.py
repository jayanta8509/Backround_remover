from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import http.client
import io
import base64
import requests
from pydantic import BaseModel, HttpUrl
from typing import Union
import logging
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    image_url: str

async def process_image(image_content: bytes, filename: str = "image.jpg") -> dict:
    """
    Process image content through PhotoRoom API
    """
    try:
        # Prepare the multipart form data
        boundary = "---011000010111000001101001"
        payload = f"""--{boundary}
Content-Disposition: form-data; name="image_file"; filename="{filename}"
Content-Type: image/jpeg

{image_content.decode('latin-1')}
--{boundary}
Content-Disposition: form-data; name="format"

jpg
--{boundary}
Content-Disposition: form-data; name="channels"

rgba
--{boundary}
Content-Disposition: form-data; name="bg_color"


--{boundary}
Content-Disposition: form-data; name="size"

hd
--{boundary}
Content-Disposition: form-data; name="crop"

false
--{boundary}
Content-Disposition: form-data; name="despill"


--{boundary}--
"""

        headers = {
            'Content-Type': f"multipart/form-data; boundary={boundary}",
            'Accept': "image/png, application/json",
            'x-api-key': "sk_pr_default_c1bd2116df663523ea1b48cfe7c9915c337837cf"
        }

        # Make request to PhotoRoom API
        conn = http.client.HTTPSConnection("sdk.photoroom.com")
        conn.request("POST", "/v1/segment", payload, headers)
        
        res = conn.getresponse()
        data = res.read()
        
        # Convert the image data to base64
        img_str = base64.b64encode(data).decode('utf-8')
        
        return {
            "status_code": 200,
            "image_base64": img_str
        }
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/api/upload-image")
async def upload_image(image: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_content = await image.read()
        return JSONResponse(content=await process_image(image_content, image.filename))
    except Exception as e:
        logger.error(f"Error in upload_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/remove-background")
async def remove_background(request: ImageRequest):
    try:
        logger.info(f"Attempting to download image from URL: {request.image_url}")
        
        # Validate URL format
        if not request.image_url.startswith(('http://', 'https://')):
            raise HTTPException(
                status_code=400, 
                detail="Invalid URL format. URL must start with http:// or https://"
            )

        # Download the image from URL with timeout and headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(
            request.image_url, 
            headers=headers,
            timeout=10,
            verify=True
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to download image. Status code: {response.status_code}")
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to download image. Status code: {response.status_code}"
            )
        
        # Check if the response is actually an image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            logger.error(f"Invalid content type: {content_type}")
            raise HTTPException(
                status_code=400,
                detail=f"URL does not point to an image. Content type: {content_type}"
            )
        
        # Process the image
        return JSONResponse(content=await process_image(response.content))
        
    except requests.exceptions.Timeout:
        logger.error("Timeout while downloading image")
        raise HTTPException(status_code=408, detail="Timeout while downloading image")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)