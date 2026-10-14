from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    status,
)

from api.predictor import predictor_service
from api.schemas import (
    PredictionItem,
    PredictionResponse,
)


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    # --------------------------------------------------------
    # Startup: load model from S3
    # --------------------------------------------------------

    predictor_service.load_model_from_s3()

    yield

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    predictor_service.model = None


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Nepali Cultural Dress Recognition API",
    description=(
        "ResNet50 Custom V4 image classification API "
        "for Nepali cultural dresses and ornaments."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Nepali Cultural Dress Recognition API",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "predict": "/predict",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
)
def health_check():

    return {
        "status": "healthy",
        "model_loaded": (
            predictor_service.model is not None
        ),
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    model_device = None

    if predictor_service.model is not None:

        model_device = str(
            next(
                predictor_service.model.parameters()
            ).device
        )

    return {
        "architecture": "ResNet50 Custom V4",
        "num_classes": predictor_service.num_classes,
        "supported_classes": predictor_service.class_names,
        "device": model_device,
        "confidence_threshold": 0.60,
        "model_info": predictor_service.checkpoint_info,
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...),
):

    # --------------------------------------------------------
    # Check whether model is loaded
    # --------------------------------------------------------

    if predictor_service.model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    # --------------------------------------------------------
    # Validate image format
    # --------------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid image format. "
                "Use JPEG, PNG, or WEBP."
            ),
        )

    # --------------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------------

    try:

        contents = await file.read()

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not read uploaded file: {exc}"
            ),
        )

    # --------------------------------------------------------
    # Check empty file
    # --------------------------------------------------------

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        predictions, is_ood = predictor_service.predict(
            contents,
            top_k=3,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )

    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return PredictionResponse(
        filename=file.filename or "unknown",
        status="uncertain" if is_ood else "success",
        top_prediction=predictions[0]["class_name"],
        confidence=predictions[0]["confidence"],
        top_k=[
            PredictionItem(**prediction)
            for prediction in predictions
        ],
        is_out_of_distribution=is_ood,
    )