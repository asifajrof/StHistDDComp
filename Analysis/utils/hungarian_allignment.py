
from utils.data_loader import DataSet, DataRecord
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

class HungarianAlignment:
    @staticmethod
    def align_datasets_labels(ref_dataset: DataSet, cmp_dataset: DataSet) -> DataSet:
        ref_labels = [record.label for record in ref_dataset.records]
        cmp_labels = [record.label for record in cmp_dataset.records]
        label_mapping = HungarianAlignment._find_best_label_mapping(ref_labels, cmp_labels)
        aligned_records = []
        for record in cmp_dataset.records:
            new_label = label_mapping.get(record.label, record.label)
            aligned_records.append(DataRecord(barcode=record.barcode, label=new_label))
        return DataSet(records=aligned_records)

    @staticmethod
    def _find_best_label_mapping(ref_labels, cmp_labels):
        contingency = pd.crosstab(pd.Series(ref_labels, name="ref"), pd.Series(cmp_labels, name="cmp"))
        print("Contingency Table:")
        print(contingency)
        cost_matrix = -contingency.values
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        label_mapping = {}
        for row, col in zip(row_ind, col_ind):
            ref_label = contingency.index[row]
            cmp_label = contingency.columns[col]
            label_mapping[cmp_label] = ref_label

        return label_mapping


if __name__ == "__main__":
    # Example usage
    ref_data = DataSet.from_csv("data/labels/conST_Melanoma_results/w_hist_hihires_swinir_large/results_seed_42/melanoma-cancer/labels.csv", barcode_index=0, label_index=1)
    cmp_data = DataSet.from_csv("data/manual_annotations/Melanoma_annotations/melanoma-cancer_annotations.csv", barcode_index=0, label_index=1)

    aligned_cmp_data = HungarianAlignment.align_datasets_labels(ref_data, cmp_data)

    print("Original Comparison Labels:")
    print([record.label for record in cmp_data.records])
    print("Aligned Comparison Labels:")
    print([record.label for record in aligned_cmp_data.records])