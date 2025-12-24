from file_extractor.model.config import ConfigModel
from file_extractor.fetch_file_by_substr import FileSearcher, SearchConfig
import json
import pandas as pd
from utils.logger import get_logger
from scipy.stats import ranksums
from statsmodels.stats.multitest import fdrcorrection
import os
from pydantic import BaseModel, Field
from utils.data_loader import DataSet, ModelComparisonData
from utils.models.model_subpath import Model_Subpath
from metric.metrics import Metrics
from file_extractor.fetch_file_by_substr import FileSearcher, SearchConfig
from tqdm import tqdm
from utils.moran_i_of_boundary_mask import MoransIOfBoundaryMask
from utils.models.cooridnate import Coordinates
from utils.boundary_discovery import BoundaryMaskDetector
logger = get_logger(__name__)


if __name__ == "__main__":
    config_path = "configs"
    search_config = SearchConfig(
        root_dir=config_path,
        substrings=[""],
        file_extension=".json",
        case_sensitive=False
    )
    file_searcher = FileSearcher(search_config)
    config_files = file_searcher.search_common()

    model_subpaths = []
    dataset_samples = set()

    for config_file in config_files:
        with open(config_file, 'r') as f:
            config_data = json.load(f)

        config = ConfigModel(**config_data)
        model_name = config.model_name
        dataset_samples = dataset_samples.union(set(config.data_samples))
        for sub_path in config.sub_paths:
            model_subpaths.append(Model_Subpath(name=model_name, sub_path=sub_path, config=config))

    dataset_samples = sorted(list(dataset_samples))

    for sample in tqdm(dataset_samples):
        columns = [f"{msp.name}_{msp.sub_path}" for msp in model_subpaths]
        columns.append("Ground_Truth")
        df_out = pd.DataFrame(index=columns,
                                        columns=["Moran's I"])
        for model_subpath_1 in model_subpaths:
            df_best_seeds = pd.read_csv(f"results/best_seeds/{model_subpath_1.name}_best_seeds.csv")
            best_seed_method_1 = df_best_seeds[(df_best_seeds['data_sample'] == sample) & (df_best_seeds['sub_path'] == model_subpath_1.sub_path)]['best_seed'].values[0]

            search_config_for_model_labels = SearchConfig(
                root_dir=model_subpath_1.config.label_base_path,
                substrings=[model_subpath_1.name, sample, f"{model_subpath_1.sub_path}/", f"/{model_subpath_1.config.seed_prefix}{str(best_seed_method_1)}"],
                file_extension=f"{model_subpath_1.config.label_file_suffix}.csv",
                case_sensitive=False
            )
            file_searcher_for_model_labels = FileSearcher(search_config_for_model_labels)
            label_files = file_searcher_for_model_labels.search_common()
            assert len(label_files) == 1, f"Expected one label file for model {model_subpath_1.name}, data sample {sample}, sub path {model_subpath_1.sub_path}, seed {best_seed_method_1}, found {len(label_files)} {label_files}"
            label_file_1 = label_files[0]
            dataset_model_1 = DataSet.from_csv(
                file_path=label_file_1,
                barcode_index=0,
                label_index=1,
            )
            
            search_config_coordinate = SearchConfig(
                root_dir=model_subpath_1.config.coordinates_base_path,
                substrings=[sample],
                file_extension=".csv",
                case_sensitive=False
            )

            file_searcher_for_coordinates = FileSearcher(search_config_coordinate)
            coordinate_files = file_searcher_for_coordinates.search_common()
            assert len(coordinate_files) == 1, f"Expected one coordinate file for sample {sample}, found {len(coordinate_files)} {coordinate_files}"
            coordinate_file = coordinate_files[0]

            dataset_coordinates = Coordinates.from_csv(
                file_path=coordinate_file,
            )

            boundary_barcodes_model_1 = BoundaryMaskDetector.detect_boundary_mask(
                coordinates=dataset_coordinates,
                data_set=dataset_model_1
            )

            morans_i_model_1 = MoransIOfBoundaryMask.calculate_morans_i(
                boundary_barcodes=boundary_barcodes_model_1,
                coordinates=dataset_coordinates
            )
            df_out.at[f"{model_subpath_1.name}_{model_subpath_1.sub_path}", "Moran's I"] = morans_i_model_1


        search_config_for_ground_truth = SearchConfig(
            root_dir=model_subpaths[0].config.annotations_base_path,
            substrings=[sample],
            file_extension=".csv",
            case_sensitive=False
        )
        file_searcher_for_ground_truth = FileSearcher(search_config_for_ground_truth)
        ground_truth_files = file_searcher_for_ground_truth.search_common()
        assert len(ground_truth_files) == 1, f"Expected one ground truth file for sample {sample}, found {len(ground_truth_files)} {ground_truth_files}"
        ground_truth_file = ground_truth_files[0]
        dataset_ground_truth = DataSet.from_csv(
            file_path=ground_truth_file,
            barcode_index=0,
            label_index=1,
        )
        boundary_barcodes_ground_truth = BoundaryMaskDetector.detect_boundary_mask(
            coordinates=dataset_coordinates,
            data_set=dataset_ground_truth
        )
        morans_i_ground_truth = MoransIOfBoundaryMask.calculate_morans_i(
            boundary_barcodes=boundary_barcodes_ground_truth,
            coordinates=dataset_coordinates
        )
        df_out.at["Ground_Truth", "Moran's I"] = morans_i_ground_truth

        output_dir = f"results/morans_i_on_boundary/{sample}/"
        os.makedirs(output_dir, exist_ok=True)
        df_out.to_csv(f"{output_dir}/morans_i_boundary_scores.csv")



