from logging import getLogger
from typing import Dict, List

from fastapi import APIRouter
from src.ml.prediction import Data, classifier

logger = getLogger(__name__)
router = APIRouter()

@router.get("/health")
def health() -> Dict[str, str]:
    return {"health": "ok"}


@router.get("/metadata")
def metadata():
    return {
        "data_type": "float32",
        "data_structure": "(1,4)",
        "data_sample": Data().data,
        "prediction_type": "float32",
        "prediction_structure": "(1,3)",
        "prediction_sample": [0.97093159, 0.01558308, 0.01348537],
    }