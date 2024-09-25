from pydantic import BaseModel, Field
from typing import List
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the possible entities appearing in the text",
    )
class Relations(BaseModel):
    """Identifying information about relationship types."""

    names: List[str] = Field(
        ...,
        description="All the possible relationship types appearing in the text",
    )