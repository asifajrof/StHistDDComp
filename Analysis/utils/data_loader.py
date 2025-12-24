
# it will be pydentic models for data validation and parsing
# all data will be a dataframe loaded from csv files in the data folder
# the dataframe will have columns: barcode, label

from pydantic import BaseModel, Field
import pandas as pd
from typing import List,Any
from utils.logger import get_logger
from utils.models.cooridnate import Coordinates
from utils.models.pcs import PcsData
from metric.metrics import Metrics
import numpy as np
logger = get_logger(__name__)

class DataRecord(BaseModel):
    barcode: int | str = Field(..., description="The barcode of the item")
    label: int | str | None = Field(None, description="The label associated with the barcode (can be None)") 

class DataSet(BaseModel):
    records: List[DataRecord] = Field(..., description="List of data records")

    @classmethod
    def from_csv(cls, file_path: str, barcode_index: int, label_index: int, index_col: int|None = None) -> "DataSet":
        #logger.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path, index_col=index_col)
        records = [
            DataRecord(
                barcode=row[barcode_index], 
                label=row[label_index] if pd.notna(row[label_index]) else None
            ) 
            for _, row in df.iterrows()
        ]
        #logger.info(f"Loaded {len(records)} records from {file_path}.")
        return cls(records=records)


# now the dataset can be from 2 model that i need to compare
# for example model1 and model2
# and i'll run validation on dataset from both models to compare their performance
# if the barcodes are not same then make the model1s barcodes the reference
# but give a slight warning if the barcodes are not same
# and there would be an argument called metric which can be ari, adjusted-mutual-info, jaccard
class ModelComparisonData(BaseModel):
    model_ref: DataSet = Field(..., description="Reference model dataset")
    model_cmp: DataSet | PcsData | Coordinates = Field(..., description="Comparison model dataset")
    metric: Metrics = Field(..., description="Metric to use for comparison")

    def calculate_metric(self) -> Any:
        self.adjust_barcodes()
        labels_true = [record.label for record in self.model_ref.records]
        if isinstance(self.model_cmp, PcsData):
            labels_pred = [record.pcs for record in self.model_cmp.pcs_list]
            labels_pred = np.array(labels_pred)
        elif isinstance(self.model_cmp, Coordinates):
            labels_pred = [[record.x,record.y] for record in self.model_cmp.points]
            labels_pred = np.array(labels_pred)
        else:
            labels_pred = [record.label for record in self.model_cmp.records]

        # Replace Nonetype labels with a "#NA" string
        labels_true = [label if label is not None else "#NA" for label in labels_true]
        labels_pred = [label if label is not None else "#NA" for label in labels_pred]
        return self.metric.calculate(labels_true, labels_pred)

    def validate_barcodes(self):
        self.model_ref.records.sort(key=lambda x: x.barcode)
        ref_barcodes = {record.barcode for record in self.model_ref.records}

        if isinstance(self.model_cmp, PcsData):
            self.model_cmp.pcs_list.sort(key=lambda x: x.barcode)
            cmp_barcodes = {record.barcode for record in self.model_cmp.pcs_list}
        elif isinstance(self.model_cmp, Coordinates):
            self.model_cmp.points.sort(key=lambda x: x.barcode)
            cmp_barcodes = {record.barcode for record in self.model_cmp.points}
        else:
            self.model_cmp.records.sort(key=lambda x: x.barcode)
            cmp_barcodes = {record.barcode for record in self.model_cmp.records}

        
        if ref_barcodes != cmp_barcodes:
            missing_in_cmp = ref_barcodes - cmp_barcodes
            missing_in_ref = cmp_barcodes - ref_barcodes
            # if the lengths are different, we can raise an error
            if len(ref_barcodes) != len(cmp_barcodes):
                raise ValueError("The number of unique barcodes in reference and comparison datasets do not match.")
            else:
                # logger.warning(f"Barcodes do not match. Missing in comparison: {missing_in_cmp}, Missing in reference: {missing_in_ref}")
                for idx, record in enumerate(self.model_ref.records):
                    if isinstance(self.model_cmp, PcsData):
                        self.model_cmp.pcs_list[idx].barcode = record.barcode
                    elif isinstance(self.model_cmp, Coordinates):
                        self.model_cmp.points[idx].barcode = record.barcode
                    else:
                        self.model_cmp.records[idx].barcode = record.barcode
                    
    def adjust_barcodes(self):
        self.validate_barcodes()
        # order the barcodes in same order as model_ref
        ref_barcode_order = {record.barcode: idx for idx, record in enumerate(self.model_ref.records)}

        if isinstance(self.model_cmp, PcsData):
            self.model_cmp.pcs_list.sort(key=lambda record: ref_barcode_order.get(record.barcode, float('inf')))
        elif isinstance(self.model_cmp, Coordinates):
            self.model_cmp.points.sort(key=lambda record: ref_barcode_order.get(record.barcode, float('inf')))
        else:
            self.model_cmp.records.sort(key=lambda record: ref_barcode_order.get(record.barcode, float('inf')))