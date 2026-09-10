from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import json
from PIL import Image
import io
import os
import uvicorn
import traceback
from typing import List, Dict, Any
import base64
import requests
from urllib.parse import urlparse

# Initialize FastAPI app
app = FastAPI(
    title="CAPTCHA Solver API",
    description="PyTorch-based API for solving 4-digit CAPTCHA images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define the CNN model architecture (must match training)
class DigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Input: 1x45x40 (Channels x Height x Width)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 3))
        self.dropout = nn.Dropout(0.4)
        
        # Flattened size: 256 channels * 4 * 3 = 3072
        self.fc1 = nn.Linear(256 * 4 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Feature Extraction Layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> 32x22x20
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> 64x11x10
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> 128x5x5
        x = F.relu(self.bn4(self.conv4(x)))             # -> 256x5x5
        
        x = self.adaptive_pool(x)                       # -> 256x4x3
        
        # Classifier Head
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    
    
# Global model variables
_model = None
_transform = None
_idx_to_class = None
_device = None
_model_loaded = False

def load_model():
    """Load the trained digit classifier model"""
    global _model, _transform, _idx_to_class, _device, _model_loaded
    
    if _model_loaded:
        return
    
    print("🔄 Loading PyTorch digit classifier model...")
    
    # Model file paths (relative to this folder)
    mapping_path = "models/class_to_idx.json"
    weights_path = "models/digit_cnn_best.pt"

    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Class mapping file not found: {mapping_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file not found: {weights_path}")
    
    # Set device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {_device}")
    
    # Load class mapping and invert it
    with open(mapping_path, 'r') as f:
        class_to_idx = json.load(f)
    _idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    print(f"📊 Classes loaded: {list(_idx_to_class.values())}")
    
    # Define transform (must match training preprocessing)
    _transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((45, 40)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # Load model
    # _model = SmallDigitCNN(num_classes=len(class_to_idx))
    _model = DigitCNN(num_classes=len(class_to_idx))
    state_dict = torch.load(weights_path, map_location=_device)
    _model.load_state_dict(state_dict)
    _model.to(_device)
    _model.eval()
    
    _model_loaded = True
    print("✅ Model loaded and ready for inference!")

def download_image_from_url(url: str, timeout: int = 10) -> Image.Image:

    # Validate URL format
    parsed_url = urlparse(url)
    if not all([parsed_url.scheme, parsed_url.netloc]):
        raise ValueError(f"Invalid URL format: {url}")
    
    # Set headers to mimic browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    print(f"📥 Downloading image from: {url}")
    
    try:
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            print(f"⚠️  Warning: Content-Type is '{content_type}', not an image type")
        
        # Check content length
        content_length = response.headers.get('content-length')
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > 10:  # Limit to 10MB
                raise ValueError(f"Image too large: {size_mb:.2f}MB (limit: 10MB)")
            print(f"📊 Image size: {size_mb:.2f}MB")
        
        # Read image data
        image_data = response.content
        print(f"📦 Downloaded {len(image_data)} bytes")
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        print(f"🖼️  Image loaded: {image.size} ({image.mode})")
        
        return image
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download image: {e}")
    except Exception as e:
        raise Exception(f"Failed to process downloaded image: {e}")

def predict_composite4_from_pil(image: Image.Image) -> tuple[str, List[float]]:
    """
    Predict 4-digit code from PIL image
    """
    global _model, _transform, _idx_to_class, _device
    
    # Ensure model is loaded
    if not _model_loaded:
        load_model()
    
    print(f"📷 Processing image of size: {image.size}")
    
    # Convert to grayscale
    img = image.convert('L')
    
    # Resize to expected composite dimensions (H=45, W=160)
    img = img.resize((160, 45))
    print(f"📏 Resized to: {img.size}")
    
    # Crop into 4 tiles: x∈[0:40], [40:80], [80:120], [120:160]
    tiles = []
    x_positions = [0, 40, 80, 120]
    
    for i, x in enumerate(x_positions):
        tile = img.crop((x, 0, x + 40, 45))
        tiles.append(tile)
        print(f"   Tile {i+1}: {x}-{x+40} → size={tile.size}")
    
    # Predict each tile
    predictions = []
    confidences = []
    
    for i, tile in enumerate(tiles):
        # Apply transforms
        img_tensor = _transform(tile).unsqueeze(0).to(_device)
        
        with torch.no_grad():
            outputs = _model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Convert to string and store
            label_str = _idx_to_class[predicted_idx.item()]
            confidence_float = confidence.item()
            
            predictions.append(label_str)
            confidences.append(confidence_float)
            print(f"   🎯 Tile {i+1}: '{label_str}' (confidence: {confidence_float:.3f})")
    
    # Concatenate predictions into final code
    concat_str = ''.join(predictions)
    print(f"🔗 Final result: '{concat_str}'")
    
    return concat_str, confidences

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load the model when the server starts"""
    try:
        load_model()
        print("🚀 FastAPI CAPTCHA Solver API is ready!")
    except Exception as e:
        print(f"❌ Failed to load model on startup: {e}")
        print(f"📍 Make sure model files exist in ../models/ directory")

# Health check endpoint
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "CAPTCHA Solver API is running!",
        "status": "healthy",
        "model_loaded": _model_loaded,
        "device": str(_device) if _device else "not_set"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if _model_loaded else "model_not_loaded",
        "model_loaded": _model_loaded,
        "device": str(_device) if _device else "not_set",
        "classes": list(_idx_to_class.values()) if _idx_to_class else []
    }

# Main CAPTCHA solving endpoint
@app.post("/solve-captcha")
async def solve_captcha(file: UploadFile = File(...)):
    """
    Solve a 4-digit CAPTCHA from an uploaded image
    
    Args:
        file: Image file (PNG, JPG, JPEG)
        
    Returns:
        JSON response with captcha digits and confidence scores
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: {file.content_type}. Expected image file."
            )
        
        print(f"📥 Received image: {file.filename} ({file.content_type})")
        
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Solve CAPTCHA
        captcha_code, confidences = predict_composite4_from_pil(image)
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "captcha_code": captcha_code,
                "digits": list(captcha_code),
                "confidences": [round(c, 4) for c in confidences],
                "average_confidence": round(avg_confidence, 4),
                "message": f"CAPTCHA solved successfully: {captcha_code}"
            }
        )
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Error solving CAPTCHA: {e}")
        print(f"📋 Stack trace: {error_details}")
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "Failed to solve CAPTCHA"
            }
        )

# Base64 image endpoint (alternative)
@app.post("/solve-captcha-base64")
async def solve_captcha_base64(request: Dict[str, Any]):
    """
    Solve a 4-digit CAPTCHA from base64 encoded image
    
    Args:
        request: JSON with 'image' field containing base64 encoded image
        
    Returns:
        JSON response with captcha digits and confidence scores
    """
    try:
        if 'image' not in request:
            raise HTTPException(
                status_code=400,
                detail="Missing 'image' field in request body"
            )
        
        base64_data = request['image']
        
        # Handle data URL format (data:image/png;base64,...)
        if base64_data.startswith('data:'):
            base64_data = base64_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        print(f"📥 Received base64 image of size: {image.size}")
        
        # Solve CAPTCHA
        captcha_code, confidences = predict_composite4_from_pil(image)
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "captcha_code": captcha_code,
                "digits": list(captcha_code),
                "confidences": [round(c, 4) for c in confidences],
                "average_confidence": round(avg_confidence, 4),
                "message": f"CAPTCHA solved successfully: {captcha_code}"
            }
        )
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Error solving base64 CAPTCHA: {e}")
        print(f"📋 Stack trace: {error_details}")
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "Failed to solve CAPTCHA from base64 data"
            }
        )

# URL-based CAPTCHA solving endpoint
@app.post("/solve-captcha-url")
async def solve_captcha_url(request: Dict[str, Any]):
    """
    Solve a 4-digit CAPTCHA from an image URL
    
    Args:
        request: JSON with 'url' field containing image URL
        
    Returns:
        JSON response with captcha digits and confidence scores
    """
    try:
        if 'url' not in request:
            raise HTTPException(
                status_code=400,
                detail="Missing 'url' field in request body"
            )
        
        image_url = request['url'].strip()
        
        if not image_url:
            raise HTTPException(
                status_code=400,
                detail="Empty URL provided"
            )
        
        print(f"🌐 Processing CAPTCHA from URL: {image_url}")
        
        # Download image from URL
        image = download_image_from_url(image_url)
        
        # Solve CAPTCHA
        captcha_code, confidences = predict_composite4_from_pil(image)
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "captcha_code": captcha_code,
                "digits": list(captcha_code),
                "confidences": [round(c, 4) for c in confidences],
                "average_confidence": round(avg_confidence, 4),
                "source_url": image_url,
                "message": f"CAPTCHA solved successfully from URL: {captcha_code}"
            }
        )
        
    except HTTPException:
        raise  # Re-raise HTTPExceptions as-is
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Error solving CAPTCHA from URL: {e}")
        print(f"📋 Stack trace: {error_details}")
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "source_url": request.get('url', 'unknown'),
                "message": "Failed to solve CAPTCHA from URL"
            }
        )

if __name__ == "__main__":
    # Fixed port: 39999 (can still be overridden by env PORT)
    port = int(os.environ.get("PORT", 3333))
    uvicorn.run(app, host="127.0.0.1", port=port)
