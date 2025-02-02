from typing import Optional, Literal
from pydantic import BaseModel

class SamplingOptions(BaseModel):
    burn_in: Optional[int] = 100
    thinning: Optional[int] = 1
    random_seed: Optional[int] = None
    cache_results: Optional[bool] = True

class SamplingRequest(BaseModel):
    method: str = "forward"
    num_samples: int = 1000
    options: SamplingOptions = SamplingOptions()

class SamplingResponseSuccess(BaseModel):
    total_samples: int
    accepted_samples: int
    rejected_samples: int
    seed_used: int

class SamplingResponseError(BaseModel):
    """Model for sampling error details"""
    error_type: str
    message: str
    details: Optional[str] = None

class SamplingResponse(BaseModel):
    """Base model for sampling response that can represent both success and failure"""
    success: bool
    error: Optional[SamplingResponseError] = None
    result: Optional[SamplingResponseSuccess] = None

class NodeDistributionRequest(BaseModel):
    node_name: str
    codomain: Literal["counts", "normalized_counts"] = "normalized_counts"

class NodeDistributionError(BaseModel):
    error_type: str
    message: str
    details: Optional[str] = None

class NodeDistributionSuccess(BaseModel):
    node_name: str
    codomain: Literal["counts", "normalized_counts"]
    x_values: list[int]
    y_values: list[int|float]

class NodeDistributionResponse(BaseModel):
    success: bool
    error: Optional[NodeDistributionError] = None
    result: Optional[NodeDistributionSuccess] = None
