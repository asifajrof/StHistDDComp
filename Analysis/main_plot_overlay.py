from file_extractor.model.config import ConfigModel
from file_extractor.fetch_file_by_substr import FileSearcher, SearchConfig
import json
import numpy as np
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
from utils.hungarian_allignment import HungarianAlignment
from utils.plot_pie_overlay import plot_pie_overlay
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
            if sub_path.endswith("_large"): continue
            model_subpaths.append(Model_Subpath(name=model_name, sub_path=sub_path, config=config))

    dataset_samples = sorted(list(dataset_samples))

    for sample in tqdm(dataset_samples):
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
        dataset_ground_truth.records.sort(key=lambda x: x.barcode)

        ground_truth_labels = [record.label for record in dataset_ground_truth.records]

        model_and_subpath_names = ['Ground_Truth']
        model_labels_alligned = [ground_truth_labels]
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

            alligned_dataset_model_1 = HungarianAlignment.align_datasets_labels(dataset_ground_truth, dataset_model_1)
            alligned_dataset_model_1.records.sort(key=lambda x: x.barcode)

            sub_path_name = model_subpath_1.sub_path
            sub_path_name = sub_path_name.replace("_hihires", "")
            sub_path_name = sub_path_name.replace("_swinir_large", "\npreprocessed")
            model_and_subpath_names.append(f"{model_subpath_1.name}\n{sub_path_name}")
            model_labels_alligned.append([record.label for record in alligned_dataset_model_1.records])

            
        search_config_coordinate = SearchConfig(
            root_dir=model_subpaths[0].config.coordinates_base_path,
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

        # sort coordinates by barcode to ensure consistent ordering
        dataset_coordinates.points.sort(key=lambda x: x.barcode)
        coordinates_pixels = [[point.x, point.y] for point in dataset_coordinates.points]
        coordinates_pixels = np.array(coordinates_pixels)
        # normalize the coordinaetes_pixels to bring them to 3000x3000 scale
        max_x = np.max(coordinates_pixels[:,0])
        max_y = np.max(coordinates_pixels[:,1])
        size_factor = 1500 if sample.lower().startswith("melanoma") else 2800
        coordinates_pixels[:,0] = coordinates_pixels[:,0] / max_x * size_factor
        coordinates_pixels[:,1] = coordinates_pixels[:,1] / max_y * size_factor

        # for model_labels_alligned, if anything is none, replace with 'Unlabeled'
        for m in range(len(model_labels_alligned)):
            model_labels_alligned[m] = [lab if lab is not None else 'Unlabeled' for lab in model_labels_alligned[m]]

        outdir = "results/plots/plots_pie_overlay"
        os.makedirs(outdir, exist_ok=True)
        plot_pie_overlay(
            labels=model_labels_alligned,
            coordinates_pixels=coordinates_pixels,
            method_names=model_and_subpath_names,
            pie_radius=15,
            pie_alpha=0.8,
            cmap_name='Set1',
            outpath=f"{outdir}/pie_overlay_{sample}_Set1.pdf"
        )


        