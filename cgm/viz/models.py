from typing import Optional, Dict, List
from pydantic import BaseModel

class SamplingOptions(BaseModel):
    burn_in: Optional[int] = 100
    thinning: Optional[int] = 1
    random_seed: Optional[int] = None
    cache_results: Optional[bool] = True

class SamplingRequest(BaseModel):
    method: str = "forward"
    num_samples: int = 1000
    conditions: Dict[str, int] = {}
    target_variable: Optional[str] = None  # Variable to get samples for
    options: SamplingOptions = SamplingOptions()

class SamplingResponse(BaseModel):
    total_samples: int
    accepted_samples: int
    rejected_samples: int
    samples: List[int]
    seed_used: int
    target_variable: str  # Name of the variable these samples are for 