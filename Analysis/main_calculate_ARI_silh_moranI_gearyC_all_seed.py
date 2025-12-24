from utils.utils import download_data
from utils.logger import get_logger
import logging
from utils.data_loader import DataSet, ModelComparisonData
from metric.metrics import Metrics
from file_extractor.fetch_file_by_substr import FileSearcher, SearchConfig
from file_extractor.model.config import ConfigModel
import glob
import pandas as pd
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
logger = get_logger(__name__)

# suppress all logging
logger.setLevel(logging.ERROR)

plt.rcParams.update({'font.size': 17, 'axes.titlesize': 17, 'axes.labelsize': 17, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16})

def parallel_process_configs(file_path):
    result_dir = "results/metrics_all_seed"
    if os.path.exists(result_dir):
        logger.info(f"Result directory {result_dir} already exists.")
        return None
    with open(file_path, "r") as f:
        json_content = f.read()
        config = ConfigModel.parse_raw(json_content)
        model = config.model_name
        logger.info(f"Processing model: {model}")

        df_results_model = pd.DataFrame()
        df_results_moran_i = pd.DataFrame()
        df_results_geary_c = pd.DataFrame()
        df_results_silhouette = pd.DataFrame()

        for sub_path in tqdm(config.sub_paths):
            for data_sample in config.data_samples:
                search_config_for_manual_annotations = SearchConfig(
                    root_dir=config.annotations_base_path,
                    substrings=[data_sample],
                    file_extension=".csv",
                    case_sensitive=False
                )
                file_searcher_for_manual_annotations = FileSearcher(search_config_for_manual_annotations)
                manual_annotation_files = file_searcher_for_manual_annotations.search_common()

                assert len(manual_annotation_files) == 1, f"Expected one manual annotation file for {data_sample}, found {len(manual_annotation_files)}"
                manual_annotation_file = manual_annotation_files[0]

                search_config_for_coordinates = SearchConfig(
                    root_dir=config.coordinates_base_path,
                    substrings=[data_sample],
                    file_extension=".csv",
                    case_sensitive=False
                )
                file_searcher_for_coordinates = FileSearcher(search_config_for_coordinates)
                coordinate_files = file_searcher_for_coordinates.search_common()

                assert len(coordinate_files) == 1, f"Expected one coordinate file for {data_sample}, found {len(coordinate_files)}"
                coordinate_file = coordinate_files[0]

                search_config_for_pcs = SearchConfig(
                    root_dir=config.pcs_base_path,
                    substrings=[data_sample],
                    file_extension=".csv",
                    case_sensitive=False
                )
                file_searcher_for_pcs = FileSearcher(search_config_for_pcs)
                pc_files = file_searcher_for_pcs.search_common()
                assert len(pc_files) == 1, f"Expected one pcs file for {data_sample}, found {len(pc_files)}"
                pc_file = pc_files[0]

                for seed in config.seeds:
                    search_config_for_model_labels = SearchConfig(
                        root_dir=config.label_base_path,
                        substrings=[model, data_sample, f"{sub_path}/", f"/{config.seed_prefix}{str(seed)}/"],
                        file_extension=f"{config.label_file_suffix}.csv",
                        case_sensitive=False
                    )
                    file_searcher_for_model_labels = FileSearcher(search_config_for_model_labels)
                    label_files = file_searcher_for_model_labels.search_common()

                    assert len(label_files) == 1, f"Expected one label file for model {model}, data sample {data_sample}, sub path {sub_path}, seed {seed}, found {len(label_files)} {label_files}"
                    
                    if not label_files:
                        logger.warning(f"No label files found for model {model}, data sample {data_sample}, sub path {sub_path}, seed {seed}. Skipping...")
                        continue

                    for label_file in label_files:
                        dataset_model = DataSet.from_csv(
                            file_path=label_file,
                            barcode_index=0,
                            label_index=1,
                        )
                        dataset_manual = DataSet.from_csv(
                            file_path=manual_annotation_file,
                            barcode_index=0,
                            label_index=1,
                        )
                        from utils.models.cooridnate import Coordinates
                        from utils.models.pcs import PcsData
                        coordinates = Coordinates.from_csv(coordinate_file)
                        pcs_data = PcsData.from_csv(pc_file)

                        comparison = ModelComparisonData(
                            model_ref=dataset_manual,
                            model_cmp=dataset_model,
                            metric=Metrics.ARI
                        )
                        ari_score = comparison.calculate_metric()

                        comparison_coords = ModelComparisonData(
                            model_ref=dataset_model,
                            model_cmp=coordinates,
                            metric=Metrics.MORANS_I
                        )
                        morans_i_score = comparison_coords.calculate_metric()

                        comparison_coords = ModelComparisonData(
                            model_ref=dataset_model,
                            model_cmp=pcs_data,
                            metric=Metrics.GEARY_C
                        )
                        geary_c_score = comparison_coords.calculate_metric()

                        comparison_pcs = ModelComparisonData(
                            model_ref=dataset_model,
                            model_cmp=pcs_data,
                            metric=Metrics.SILHOUETTE
                        )
                        silhouette_score = comparison_pcs.calculate_metric()

                        df_results_model = pd.concat([
                            df_results_model,
                            pd.DataFrame([{
                                "model": model,
                                "data_sample": data_sample,
                                "sub_path": sub_path,
                                "seed": seed,
                                "metric": Metrics.ARI.name,
                                "score": ari_score
                            }])
                        ], ignore_index=True)

                        df_results_moran_i = pd.concat([
                            df_results_moran_i,
                            pd.DataFrame([{
                                "model": model,
                                "data_sample": data_sample,
                                "sub_path": sub_path,
                                "seed": seed,
                                "metric": Metrics.MORANS_I.name,
                                "score": morans_i_score
                            }])
                        ], ignore_index=True)
                        df_results_geary_c = pd.concat([
                            df_results_geary_c,
                            pd.DataFrame([{
                                "model": model,
                                "data_sample": data_sample,
                                "sub_path": sub_path,
                                "seed": seed,
                                "metric": Metrics.GEARY_C.name,
                                "score": geary_c_score
                            }])
                        ], ignore_index=True)
                        df_results_silhouette = pd.concat([
                            df_results_silhouette,
                            pd.DataFrame([{
                                "model": model,
                                "data_sample": data_sample,
                                "sub_path": sub_path,
                                "seed": seed,
                                "metric": Metrics.SILHOUETTE.name,
                                "score": silhouette_score
                            }])
                        ], ignore_index=True)

    os.makedirs(result_dir, exist_ok=True)
    results_file = os.path.join(result_dir, f"{model}_ari.csv")
    df_results_model.to_csv(results_file, index=False)
    logger.info(f"Model comparison results saved to {results_file}.")
    

    silhouette_results_file = os.path.join(result_dir, f"{model}_silhouette.csv")
    df_results_silhouette.to_csv(silhouette_results_file, index=False)
    logger.info(f"Silhouette comparison results saved to {silhouette_results_file}.")
    moran_i_results_file = os.path.join(result_dir, f"{model}_moran_i.csv")
    df_results_moran_i.to_csv(moran_i_results_file, index=False)
    logger.info(f"Moran's I comparison results saved to {moran_i_results_file}.")
    geary_c_results_file = os.path.join(result_dir, f"{model}_geary_c.csv")
    df_results_geary_c.to_csv(geary_c_results_file, index=False)
    logger.info(f"Geary's C comparison results saved to {geary_c_results_file}.")
    return True
if __name__ == "__main__":

    logger.info("Starting the data download process...")
    download_data()

    config_path = "configs"
    search_config = SearchConfig(
        root_dir=config_path,
        substrings=[""],
        file_extension=".json",
        case_sensitive=False
    )
    file_searcher = FileSearcher(search_config)
    config_files = file_searcher.search_common()
    print(f"Found {len(config_files)} configuration files for processing.")

    from utils.parallel_executor import ParallelExecutor
    parallel_executor = ParallelExecutor(func=parallel_process_configs, configs=config_files)
    results = parallel_executor.execute()

    method = [
        'conST',
        'SpaGCN',
        'stLearn',
        'DeepST',
        'ScribbleDom'
    ]

    samples = ["151507","151508","151509","151510","151669","151670","151671","151672","151673","151674","151675","151676","hbc","bcdc","melanoma"]

    dfs = []
    for m in method:
        df = pd.read_csv(f'results/metrics_all_seed/{m}_ari.csv')
        dfs.append(df)
    df_all_data_ari = pd.concat(dfs, ignore_index=True)
    df_all_data_ari = df_all_data_ari[~df_all_data_ari['sub_path'].str.contains('w_hist_hihires_swinir_large')]

    dfs = []
    for m in method:
        df = pd.read_csv(f'results/metrics_all_seed/{m}_moran_i.csv')
        dfs.append(df)
    df_all_data_moran_i = pd.concat(dfs, ignore_index=True)
    df_all_data_moran_i = df_all_data_moran_i[~df_all_data_moran_i['sub_path'].str.contains('w_hist_hihires_swinir_large')]

    dfs = []
    for m in method:
        df = pd.read_csv(f'results/metrics_all_seed/{m}_geary_c.csv')
        dfs.append(df)
    df_all_data_geary_c = pd.concat(dfs, ignore_index=True)
    df_all_data_geary_c = df_all_data_geary_c[~df_all_data_geary_c['sub_path'].str.contains('w_hist_hihires_swinir_large')]

    dfs = []
    for m in method:
        df = pd.read_csv(f'results/metrics_all_seed/{m}_silhouette.csv')
        dfs.append(df)
    df_all_data_silhouette = pd.concat(dfs, ignore_index=True)
    df_all_data_silhouette = df_all_data_silhouette[~df_all_data_silhouette['sub_path'].str.contains('w_hist_hihires_swinir_large')]
    # seaborn distribution plot violin plot for ARI scores
    for sample in samples:
        df_all_data_ari_sample = df_all_data_ari[df_all_data_ari['data_sample'] == sample]
        df_all_data_ari_sample['histology_status'] = df_all_data_ari_sample['sub_path'].apply(lambda x: 'With Histology/Scribble' if x in ['w_hist_hihires', 'expert'] else 'Without Histology')
        fig = plt.figure(figsize=(10, 6))
        sns.violinplot(x='model', y='score', data=df_all_data_ari_sample, inner='point', hue='histology_status', split=True)
        plt.title(f'ARI Score Distribution by Histology Status for Sample {sample}')
        plt.ylabel('ARI Score')
        plt.xlabel('Method')
        plt.ylim(0, 1)
        plt.grid(axis='y')

        plt.tight_layout()
        out_dir = 'results/plots/metrices_all_seed/ari'
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, f'ari_violinplot_{sample}.pdf')
        if file_path.endswith('.eps'):
            plt.savefig(file_path, format='eps', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

    # seaborn distribution plot violin plot for Moran's I scores
    for sample in samples:
        df_all_data_moran_i_sample = df_all_data_moran_i[df_all_data_moran_i['data_sample'] == sample]
        df_all_data_moran_i_sample['histology_status'] = df_all_data_moran_i_sample['sub_path'].apply(lambda x: 'With Histology/Scribble' if x in ['w_hist_hihires', 'expert'] else 'Without Histology')
        fig = plt.figure(figsize=(10, 6))
        sns.violinplot(x='model', y='score', data=df_all_data_moran_i_sample, inner='point', hue='histology_status', split=True)
        plt.title(f"Moran's I Score Distribution by Histology Status for Sample {sample}")
        plt.ylabel("Moran's I Score")
        plt.xlabel('Method')
        plt.ylim(-1, 1)
        plt.grid(axis='y')

        plt.tight_layout()
        out_dir = 'results/plots/metrices_all_seed/morans_i'
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, f'morans_i_violinplot_{sample}.pdf')
        if file_path.endswith('.eps'):
            plt.savefig(file_path, format='eps', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

    # seaborn distribution plot violin plot for Geary's C scores
    for sample in samples:
        df_all_data_geary_c_sample = df_all_data_geary_c[df_all_data_geary_c['data_sample'] == sample]
        df_all_data_geary_c_sample['histology_status'] = df_all_data_geary_c_sample['sub_path'].apply(lambda x: 'With Histology/Scribble' if x in ['w_hist_hihires', 'expert'] else 'Without Histology')
        fig = plt.figure(figsize=(10, 6))
        sns.violinplot(x='model', y='score', data=df_all_data_geary_c_sample, inner='point', hue='histology_status', split=True)
        plt.title(f"Geary's C Score Distribution by Histology Status for Sample {sample}")
        plt.ylabel("Geary's C Score")
        plt.xlabel('Method')
        plt.ylim(0, 2)
        plt.grid(axis='y')

        plt.tight_layout()
        out_dir = 'results/plots/metrices_all_seed/gearys_c'
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, f'gearys_c_violinplot_{sample}.pdf')
        if file_path.endswith('.eps'):
            plt.savefig(file_path, format='eps', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
    # seaborn distribution plot violin plot for Silhouette scores
    for sample in samples:
        df_all_data_silhouette_sample = df_all_data_silhouette[df_all_data_silhouette['data_sample'] == sample]
        df_all_data_silhouette_sample['histology_status'] = df_all_data_silhouette_sample['sub_path'].apply(lambda x: 'With Histology/Scribble' if x in ['w_hist_hihires', 'expert'] else 'Without Histology')
        fig = plt.figure(figsize=(10, 6))
        sns.violinplot(x='model', y='score', data=df_all_data_silhouette_sample, inner='point', hue='histology_status', split=True)
        plt.title(f'Silhouette Score Distribution by Histology Status for Sample {sample}')
        plt.ylabel('Silhouette Score')
        plt.xlabel('Method')
        plt.ylim(-1, 1)
        plt.grid(axis='y')

        plt.tight_layout()
        out_dir = 'results/plots/metrices_all_seed/silhouette'
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, f'silhouette_violinplot_{sample}.pdf')
        if file_path.endswith('.eps'):
            plt.savefig(file_path, format='eps', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()