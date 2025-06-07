from pydantic import BaseModel, Field


class MissionSpec(BaseModel):
    bus_u: int = Field(ge=1, le=12, description="CubeSat units (1U = 10×10×10 cm)")
    payload_mass_kg: float
    orbit_alt_km: float
    mass_limit_kg: float = 12.0
