from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.predictor import predictor_service
from api.schemas import PredictionItem, PredictionResponse


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the existing trained checkpoint once when the API starts.
    predictor_service.load_model_from_s3()
    yield
    predictor_service.model = None


app = FastAPI(
    title="Nepali Cultural Dress Recognition API",
    description="ResNet50 Custom V4 image classification for Nepali cultural attire.",
    version="1.1.0",
    lifespan=lifespan,
)

# Serve the web interface and its assets from the same origin as the API.
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "healthy",
        "model_loaded": predictor_service.model is not None,
    }


@app.get("/model-info")
def model_info():
    model_device = None
    if predictor_service.model is not None:
        model_device = str(next(predictor_service.model.parameters()).device)

    return {
        "architecture": "ResNet50 Custom V4",
        "num_classes": predictor_service.num_classes,
        "supported_classes": predictor_service.class_names,
        "device": model_device,
        "confidence_threshold": 0.60,
        "model_info": predictor_service.checkpoint_info,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if predictor_service.model is None:
        raise HTTPException(status_code=503, detail="The model is not loaded yet.")

    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Please upload a JPG, PNG, or WEBP image.",
        )

    try:
        contents = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.") from exc
    finally:
        await file.close()

    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # Keep the frontend's upload limit aligned with the model service's 5 MB limit.
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum size is 5 MB.")

    try:
        predictions, is_ood = predictor_service.predict(contents, top_k=3)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="The model is not ready to predict.") from exc
    except Exception as exc:
        # Log details in server logs, but do not expose internals to the browser.
        print(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail="Prediction failed. Please try another image.") from exc

    return PredictionResponse(
        filename=file.filename or "uploaded-image",
        status="uncertain" if is_ood else "success",
        top_prediction=predictions[0]["class_name"],
        confidence=predictions[0]["confidence"],
        top_k=[PredictionItem(**item) for item in predictions],
        is_out_of_distribution=is_ood,
    )
