from pydantic import BaseModel, Field

class ElectionPredictionRequest(BaseModel):
    dmk_strength: float = Field(..., ge=0, le=100, description="Strength of DMK (0-100)")
    admk_strength: float = Field(..., ge=0, le=100, description="Strength of ADMK (0-100)")
    tvk_strength: float = Field(..., ge=0, le=100, description="Strength of TVK (0-100)")
    ntk_strength: float = Field(..., ge=0, le=100, description="Strength of NTK (0-100)")
    pmk_strength: float = Field(..., ge=0, le=100, description="Strength of PMK (0-100)")
    bjp_strength: float = Field(..., ge=0, le=100, description="Strength of BJP (0-100)")
    inc_strength: float = Field(..., ge=0, le=100, description="Strength of INC (0-100)")
    literacy_rate: float = Field(..., ge=0, le=100, description="Literacy rate of the constituency (0-100)")
    urban_population_percentage: float = Field(..., ge=0, le=100, description="Urban population percentage (0-100)")
    anti_incumbency_factor: float = Field(..., ge=0, le=10, description="Anti-incumbency factor (0-10)")
    campaign_spending_index: float = Field(..., ge=1, le=10, description="Campaign spending index (1-10)")

    class Config:
        json_schema_extra = {
            "example": {
                "dmk_strength": 65.5,
                "admk_strength": 60.0,
                "tvk_strength": 35.0,
                "ntk_strength": 20.0,
                "pmk_strength": 15.0,
                "bjp_strength": 10.0,
                "inc_strength": 12.0,
                "literacy_rate": 80.5,
                "urban_population_percentage": 50.0,
                "anti_incumbency_factor": 7.5,
                "campaign_spending_index": 8.0
            }
        }

class ElectionPredictionResponse(BaseModel):
    prediction: int = Field(..., description="0: DMK, 1: ADMK, 2: TVK, 3: NTK, 4: PMK, 5: BJP, 6: INC")
    prediction_label: str = Field(..., description="String label for the prediction")
    probability_dmk: float = Field(..., description="Probability of DMK winning")
    probability_admk: float = Field(..., description="Probability of ADMK winning")
    probability_tvk: float = Field(..., description="Probability of TVK winning")
    probability_ntk: float = Field(..., description="Probability of NTK winning")
    probability_pmk: float = Field(..., description="Probability of PMK winning")
    probability_bjp: float = Field(..., description="Probability of BJP winning")
    probability_inc: float = Field(..., description="Probability of INC winning")
