from file_extractor.model.config import ConfigModel
from file_extractor.fetch_file_by_substr import FileSearcher, SearchConfig
import json
import pandas as pd
from utils.logger import get_logger
from scipy.stats import ranksums
from statsmodels.stats.multitest import fdrcorrection
import os
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

    for config_file in config_files:
        logger.info(f"Processing configuration file: {config_file}")
        with open(config_file, "r") as f:
            config_data = json.load(f)
        
        config = ConfigModel(**config_data)

        ari_file = f"results/ari/{config.model_name}.csv"
        
        df_results_model = pd.read_csv(ari_file)

        df = pd.DataFrame()
        for data_sample in config.data_samples:
            for sub_path in config.sub_paths:
                df_subset = df_results_model[(df_results_model['sub_path'] == sub_path) & (df_results_model['data_sample'] == data_sample)]
                if df_subset.empty:
                    logger.warning(f"No results found for model: {config.model_name}, data sample: {data_sample}, sub path: {sub_path}")
                    continue
                best_row = df_subset.loc[df_subset['score'].idxmax()]
                logger.info(f"Best score for model: {config.model_name}, data sample: {data_sample}, sub path: {sub_path} is {best_row['score']} with seed {best_row['seed']}")
                df = pd.concat([df, pd.DataFrame({
                    'model_name': [config.model_name],
                    'data_sample': [data_sample],
                    'sub_path': [sub_path],
                    'best_score': [best_row['score']],
                    'best_seed': [best_row['seed']]
                })], ignore_index=True)

        output_dir = "results/best_seeds"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{config.model_name}_best_seeds.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Best seeds results saved to {output_file}")