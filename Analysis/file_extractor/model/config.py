from pydantic import BaseModel
from typing import List
from pydantic import Field
import json

class ConfigModel(BaseModel):
    annotations_base_path: str = Field("", description="Base path for annotations")
    coordinates_base_path: str = Field("", description="Base path for coordinates")
    pcs_base_path: str = Field("", description="Base path for pcs")
    label_base_path: str = Field("", description="Base path for labels")
    model_name: str = Field("", description="Name of the model")
    sub_paths: List[str] = Field(default=["", "/default/subpath2"], description="List of sub paths")
    wo_hist_path: str = Field("", description="Path for wo_hist")
    seeds: List[int] = Field(default=[], description="List of seed values")
    data_samples: List[str] = Field(default=[], description="List of data samples")
    seed_prefix: str = Field("", description="Prefix for seed directories")
    label_file_suffix: str = Field("labels", description="Suffix for label files")

if __name__ == "__main__":
    # read the config from json file
    with open("configs/conST/config.json", "r") as f:
        config_data = json.load(f)
    
    config = ConfigModel(**config_data)
    print(config)