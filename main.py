from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import onnxruntime as ort
import numpy as np
from PIL import Image
import json
import io
import os
import uvicorn
import traceback
from typing import List, Dict, Any, Tuple
import base64
import requests
from urllib.parse import urlparse

# Global runtime variables
_session = None
_idx_to_class = None
_model_loaded = False
MODEL_PATH = "models/digit_cnn.onnx"
MAPPING_PATH = "models/class_to_idx.json"


def load_model():
    """Load the trained digit classifier ONNX model"""
    global _session, _idx_to_class, _model_loaded

    if _model_loaded:
        return

    print("🔄 Loading ONNX digit classifier model...")

    if not os.path.exists(MAPPING_PATH):
        raise FileNotFoundError(f"Class mapping file not found: {MAPPING_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"ONNX model file not found: {MODEL_PATH}")

    # Load class mapping and invert it
    with open(MAPPING_PATH, "r") as f:
        class_to_idx = json.load(f)
    _idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    print(f"📊 Classes loaded: {list(_idx_to_class.values())}")

    # Initialize ONNX Runtime session with CPU provider
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 2

    _session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=session_options,
        providers=["CPUExecutionProvider"]
    )

    _model_loaded = True
    print("✅ ONNX Model loaded and ready for ultra-fast inference!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler for model initialization and cleanup"""
    try:
        load_model()
        print("🚀 FastAPI CAPTCHA Solver API (ONNX Runtime) is ready!")
    except Exception as e:
        print(f"❌ Failed to load model on startup: {e}")
        print("📍 Make sure model files exist in models/ directory")
    yield


# Initialize FastAPI app
app = FastAPI(
    title="CAPTCHA Solver API (ONNX)",
    description="Ultra-fast ONNX Runtime API for solving 4-digit CAPTCHA images",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def download_image_from_url(url: str, timeout: int = 10) -> Image.Image:
    """Download and validate an image from an external URL"""
    parsed_url = urlparse(url)
    if not all([parsed_url.scheme, parsed_url.netloc]):
        raise ValueError(f"Invalid URL format: {url}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        ),
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    print(f"📥 Downloading image from: {url}")

    try:
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if not content_type.startswith("image/"):
            print(f"⚠️  Warning: Content-Type is '{content_type}', not an image type")

        content_length = response.headers.get("content-length")
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > 10:
                raise ValueError(f"Image too large: {size_mb:.2f}MB (limit: 10MB)")
            print(f"📊 Image size: {size_mb:.2f}MB")

        image = Image.open(io.BytesIO(response.content))
        print(f"🖼️  Image loaded: {image.size} ({image.mode})")
        return image

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download image: {e}")
    except Exception as e:
        raise Exception(f"Failed to process downloaded image: {e}")


def predict_composite4_from_pil(image: Image.Image) -> Tuple[str, List[float]]:
    """
    Predict 4-digit code from PIL image using batched ONNX Runtime inference
    """
    global _session, _idx_to_class, _model_loaded

    if not _model_loaded:
        load_model()

    print(f"📷 Processing image of size: {image.size}")

    # Convert to grayscale and resize to expected composite dimensions (H=45, W=160)
    img = image.convert("L").resize((160, 45))
    print(f"📏 Resized composite to: {img.size}")

    # Crop into 4 tiles: x in [0:40], [40:80], [80:120], [120:160]
    x_positions = [0, 40, 80, 120]
    tile_arrays = []

    for i, x in enumerate(x_positions):
        tile = img.crop((x, 0, x + 40, 45))
        # PIL resize takes (width, height) -> (40, 45)
        tile_resized = tile.resize((40, 45))
        # Pure NumPy normalization: scales [0, 255] -> [-1.0, 1.0] (matching PyTorch (x - 0.5) / 0.5)
        normalized = (np.array(tile_resized, dtype=np.float32) / 255.0 - 0.5) / 0.5
        tile_arrays.append(normalized)
        print(f"   Tile {i+1}: {x}-{x+40} → size={tile.size}")

    # Stack into single batch tensor: shape [4, 1, 45, 40]
    batch_input = np.stack(tile_arrays, axis=0)[:, np.newaxis, :, :]

    # Single batched forward pass through ONNX Runtime (< 1ms execution)
    outputs = _session.run(None, {"input": batch_input})[0]  # shape [4, 10]

    # Compute numerical-stable Softmax across digit classes
    exp_logits = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    predicted_indices = np.argmax(probabilities, axis=1)
    confidence_values = np.max(probabilities, axis=1)

    predictions = []
    confidences = []

    for i in range(len(predicted_indices)):
        digit = _idx_to_class[int(predicted_indices[i])]
        conf = float(confidence_values[i])
        predictions.append(digit)
        confidences.append(conf)
        print(f"   🎯 Tile {i+1}: '{digit}' (confidence: {conf:.3f})")

    concat_str = "".join(predictions)
    print(f"🔗 Final result: '{concat_str}'")

    return concat_str, confidences


# Health check endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "CAPTCHA Solver API (ONNX Runtime) is running!",
        "status": "healthy",
        "runtime": "onnxruntime",
        "model_loaded": _model_loaded,
        "device": "cpu",
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if _model_loaded else "model_not_loaded",
        "runtime": "onnxruntime",
        "model_loaded": _model_loaded,
        "model_path": MODEL_PATH,
        "device": "cpu",
        "classes": list(_idx_to_class.values()) if _idx_to_class else [],
    }


# Main CAPTCHA solving endpoint (Multipart File)
@app.post("/solve-captcha")
async def solve_captcha(file: UploadFile = File(...)):
    """
    Solve a 4-digit CAPTCHA from an uploaded image
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Expected image file.",
            )

        print(f"📥 Received image: {file.filename} ({file.content_type})")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        captcha_code, confidences = predict_composite4_from_pil(image)
        avg_confidence = sum(confidences) / len(confidences)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "captcha_code": captcha_code,
                "digits": list(captcha_code),
                "confidences": [round(c, 4) for c in confidences],
                "average_confidence": round(avg_confidence, 4),
                "message": f"CAPTCHA solved successfully: {captcha_code}",
            },
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
                "message": "Failed to solve CAPTCHA",
            },
        )


# Base64 image endpoint (used by Tampermonkey userscript)
@app.post("/solve-captcha-base64")
async def solve_captcha_base64(request: Dict[str, Any]):
    """
    Solve a 4-digit CAPTCHA from base64 encoded image
    """
    try:
        if "image" not in request:
            raise HTTPException(
                status_code=400, detail="Missing 'image' field in request body"
            )

        base64_data = request["image"]

        # Handle data URL format (data:image/png;base64,...)
        if base64_data.startswith("data:"):
            base64_data = base64_data.split(",")[1]

        image_bytes = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_bytes))

        print(f"📥 Received base64 image of size: {image.size}")

        captcha_code, confidences = predict_composite4_from_pil(image)
        avg_confidence = sum(confidences) / len(confidences)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "captcha_code": captcha_code,
                "digits": list(captcha_code),
                "confidences": [round(c, 4) for c in confidences],
                "average_confidence": round(avg_confidence, 4),
                "message": f"CAPTCHA solved successfully: {captcha_code}",
            },
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
                "message": "Failed to solve CAPTCHA from base64 data",
            },
        )


# URL-based CAPTCHA solving endpoint
@app.post("/solve-captcha-url")
async def solve_captcha_url(request: Dict[str, Any]):
    """
    Solve a 4-digit CAPTCHA from an image URL
    """
    try:
        if "url" not in request:
            raise HTTPException(
                status_code=400, detail="Missing 'url' field in request body"
            )

        image_url = request["url"].strip()
        if not image_url:
            raise HTTPException(status_code=400, detail="Empty URL provided")

        print(f"🌐 Processing CAPTCHA from URL: {image_url}")

        image = download_image_from_url(image_url)
        captcha_code, confidences = predict_composite4_from_pil(image)
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
                "message": f"CAPTCHA solved successfully from URL: {captcha_code}",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Error solving CAPTCHA from URL: {e}")
        print(f"📋 Stack trace: {error_details}")

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "source_url": request.get("url", "unknown"),
                "message": "Failed to solve CAPTCHA from URL",
            },
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3333))
    uvicorn.run(app, host="127.0.0.1", port=port)
