from pydantic import BaseModel, Field

class Pcs(BaseModel):
    barcode: str = Field(..., description="Barcode identifier")
    pcs: list[float] = Field(..., description="List of principal components")

class PcsData(BaseModel):
    pcs_list: list[Pcs] = Field(..., description="List of Pcs objects")

    @classmethod
    def from_csv(cls, file_path: str) -> "PcsData":
        import pandas as pd
        df = pd.read_csv(file_path)
        pcs_list = []
        for index, row in df.iterrows():
            barcode = row[0]
            pcs = row[1:].tolist()
            pcs_list.append(Pcs(barcode=barcode, pcs=pcs))

        pcs_list.sort(key=lambda x: x.barcode)
        return cls(pcs_list=pcs_list)