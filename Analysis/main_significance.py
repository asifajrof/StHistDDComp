from file_extractor.model.config import ConfigModel
from file_extractor.fetch_file_by_substr import FileSearcher, SearchConfig
import json
import pandas as pd
from utils.logger import get_logger
from scipy.stats import ranksums
from statsmodels.stats.multitest import fdrcorrection
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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
        
        ideal_sub = config.wo_hist_path
        other_subs = [sub for sub in config.sub_paths if sub != ideal_sub]

        if ideal_sub:
            for other_sub in other_subs:
                df = pd.DataFrame()
                p_values = []
                data_samples = []

                for data_sample in config.data_samples:                    
                    df_ideal = df_results_model[(df_results_model['sub_path'] == ideal_sub) & (df_results_model['data_sample'] == data_sample)]
                    df_other = df_results_model[(df_results_model['sub_path'] == other_sub) & (df_results_model['data_sample'] == data_sample)]

                    stat, p = ranksums(df_other['score'].values, df_ideal['score'].values, alternative='greater')
                    p_values.append(p)
                    data_samples.append(data_sample)

                # Apply FDR correction
                _, p_values_corrected = fdrcorrection(p_values, alpha=0.05)

                for data_sample, p, p_corrected in zip(data_samples, p_values, p_values_corrected):
                    df = pd.concat([df, pd.DataFrame({
                        'model_name': [config.model_name],
                        'data_sample': [data_sample],
                        'ideal_sub_path': [ideal_sub],
                        'other_sub_path': [other_sub],
                        'p_value': [p],
                        'p_value_corrected': [p_corrected]
                    })], ignore_index=True)

                output_dir = "results/significance"
                os.makedirs(output_dir, exist_ok=True)
                output_file = f"{output_dir}/{config.model_name}_significance_{ideal_sub}_vs_{other_sub}.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Significance results saved to {output_file}")

    df_const_w_hist_preprocessed = pd.read_csv("results/significance/conST_significance_wo_hist_vs_w_hist_hihires_swinir_large.csv")
    df_const_w_hist = pd.read_csv("results/significance/conST_significance_wo_hist_vs_w_hist_hihires.csv")
    df_SpaGCN_w_hist_preprocessed = pd.read_csv("results/significance/SpaGCN_significance_wo_hist_vs_w_hist_hihires_swinir_large.csv")
    df_SpaGCN_w_hist = pd.read_csv("results/significance/SpaGCN_significance_wo_hist_vs_w_hist_hihires.csv")
    df_scribbledom_w_scribble = pd.read_csv("results/significance/ScribbleDom_significance_mclust_vs_expert.csv")
    df_const_w_hist = df_const_w_hist[['data_sample', 'p_value_corrected']]
    df_const_w_hist_preprocessed = df_const_w_hist_preprocessed[['data_sample', 'p_value_corrected']]
    df_SpaGCN_w_hist = df_SpaGCN_w_hist[['data_sample', 'p_value_corrected']]
    df_SpaGCN_w_hist_preprocessed = df_SpaGCN_w_hist_preprocessed[['data_sample', 'p_value_corrected']]
    df_scribbledom_w_scribble = df_scribbledom_w_scribble[['data_sample', 'p_value_corrected']]
    df_merged_by_data_sample = df_const_w_hist.merge(
        df_const_w_hist_preprocessed, on='data_sample', suffixes=('_const_w_hist', '_const_w_hist_preprocessed')
    ).merge(
        df_SpaGCN_w_hist.rename(columns={'p_value_corrected': 'p_value_corrected_SpaGCN_w_hist'}), on='data_sample'
    ).merge(
        df_SpaGCN_w_hist_preprocessed.rename(columns={'p_value_corrected': 'p_value_corrected_SpaGCN_w_hist_preprocessed'}), on='data_sample'
    ).merge(
        df_scribbledom_w_scribble.rename(columns={'p_value_corrected': 'p_value_corrected_scribbledom_w_scribble'}), on='data_sample'
    )
    df_merged_by_data_sample.set_index('data_sample', inplace=True)
    # neg log and round to 3 decimal places
    df_neg_log = -np.log10(df_merged_by_data_sample)
    df_neg_log = df_neg_log.round(2)
    df_neg_log.columns

    columns = [
        'const\nw_hist',
        'const\nw_hist\npreprocessed',
        'SpaGCN\nw_hist',
        'SpaGCN\nw_hist\npreprocessed',
        'ScribbleDom\nw_scribble'
    ]
    df_neg_log.columns = columns
    # plot heatmap sns
    fig = plt.figure(figsize=(10, 6))

    sns.heatmap(df_neg_log, annot=True, cmap='magma', cbar_kws={'label': '-log10(p-value-corrected)'})
    plt.title('Significance of Histology Information Across Methods and Datasets')
    plt.ylabel('Data Sample')
    plt.xlabel('Method and Condition')
    plt.tight_layout()
    outpath = 'results/plots/significance/heatmap_significance_methods_datasets.pdf'
    if outpath.endswith('.eps'):
        fig.savefig(outpath, format='eps', dpi=300, bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig(outpath, dpi=300, bbox_inches='tight', pad_inches=0.1)

    plt.show()
    plt.close(fig)