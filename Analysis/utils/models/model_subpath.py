from pydantic import BaseModel, Field
from file_extractor.model.config import ConfigModel

class Model_Subpath(BaseModel):
    name: str
    sub_path: str
    config: ConfigModel = Field(default=None)