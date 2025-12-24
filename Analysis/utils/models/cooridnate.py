
from pydantic import BaseModel, Field
import pandas as pd

class Coordinate(BaseModel):
    barcode: str = Field(..., description="Barcode identifier")
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")

class Coordinates(BaseModel):
    points: list[Coordinate] = Field(..., description="List of coordinates")

    @staticmethod
    def from_csv(file_path: str) -> "Coordinates":
        # 1st column is barcode and 2nd and 3rd columns are x and y coordinates
        df = pd.read_csv(file_path)
        points = [Coordinate(barcode=row[0], x=row[1], y=row[2]) for index, row in df.iterrows()]
        return Coordinates(points=points)