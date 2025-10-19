from pydantic import BaseModel
from typing import Dict, Optional, Any

class OrderData(BaseModel):
    distance_in_meters: float
    duration_in_seconds: float
    pickup_in_meters: float
    pickup_in_seconds: float
    price_start_local: float
    driver_rating: float = 5.0
    carname: str = "unknown"
    carmodel: str = "unknown"
    platform: str = "unknown"
    order_timestamp: Optional[str] = None
    current_price: Optional[float] = None

class OptimizationResponse(BaseModel):
    success: bool
    strategies: Dict[str, Any]
    message: Optional[str] = None

class TrainingResponse(BaseModel):
    success: bool
    message: str
    metrics: Optional[Dict[str, Any]] = None