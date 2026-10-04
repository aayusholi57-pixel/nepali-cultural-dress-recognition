from typing import List

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    class_name: str
    confidence: float = Field(
        ...,
        description="Prediction confidence as a percentage."
    )


class PredictionResponse(BaseModel):
    filename: str
    status: str
    top_prediction: str
    confidence: float
    top_k: List[PredictionItem]
    is_out_of_distribution: bool