import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# API endpoint
url = "http://localhost:8000/predict"

# Image URL to process - this can be changed by the user
image_url = "https://i-family.info/wp-content/uploads/2024/01/Eheringe-678x381.jpeg"

# Send the request to the API
response = requests.post(url=url, json={"input": image_url})

# Check if the request was successful
if response.status_code == 200:
    # Open the image from the response content
    img = Image.open(BytesIO(response.content))
    
    # Display the processed image
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title('Processed Image with Transparent Background')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Save the processed image
    img.save("output_image_transparent.png")
    print("Image processed successfully and saved as output_image_transparent.png")
else:
    print(f"Error: {response.status_code}")
    print(response.text)