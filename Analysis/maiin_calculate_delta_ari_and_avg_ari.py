import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

method = [
    'conST',
    'SpaGCN',
    # 'stLearn',
    # 'DeepST',
    'ScribbleDom'
]

samples = ["151507","151508","151509","151510","151673","151674","151675","151676","151669","151670","151671","151672","hbc","bcdc","melanoma"]

dfs = []
for m in method:
    df = pd.read_csv(f'./results/metrics_all_seed/{m}_ari.csv', index_col=0)
    df['method'] = m
    dfs.append(df)
result_df = pd.concat(dfs, ignore_index=True)

df_out = pd.DataFrame(
    index=samples,
    columns=['delta_ari_with_hist_conST', 'delta_ari_with_hist_preprocessed_conST', 'ari_base_without_hist_conST', 'delta_ari_with_hist_SpaGCN', 'delta_ari_with_hist_preprocessed_SpaGCN',
             'ari_base_without_hist_SpaGCN','delta_ari_with_scribble_ScribbleDom', 'ari_base_without_hist_ScribbleDom']
)

for sample in samples:
    # fill out df_out
    for m in method:
        if m in ['conST', 'SpaGCN']:
            ari_with_hist = result_df[(result_df['method'] == m) & (result_df['data_sample'] == sample) & (result_df['sub_path'] == 'w_hist_hihires')]['score'].mean()
            ari_with_hist_preprocessed = result_df[(result_df['method'] == m) & (result_df['data_sample'] == sample) & (result_df['sub_path'] == 'w_hist_hihires_swinir_large')]['score'].mean()
            ari_without_hist = result_df[(result_df['method'] == m) & (result_df['data_sample'] == sample) & (result_df['sub_path'] == 'wo_hist')]['score'].mean()
            delta_ari_with_hist = ari_with_hist - ari_without_hist
            delta_ari_with_hist_preprocessed = ari_with_hist_preprocessed - ari_without_hist
            df_out.loc[sample, 'ari_base_without_hist_' + m] = ari_without_hist
            df_out.loc[sample, 'delta_ari_with_hist_' + m] = delta_ari_with_hist
            df_out.loc[sample, 'delta_ari_with_hist_preprocessed_' + m] = delta_ari_with_hist_preprocessed
        elif m == 'ScribbleDom':
            ari_with_scribble = result_df[(result_df['method'] == m) & (result_df['data_sample'] == sample) & (result_df['sub_path'] == 'expert')]['score'].mean()
            ari_without_scribble = result_df[(result_df['method'] == m) & (result_df['data_sample'] == sample) & (result_df['sub_path'] == 'mclust')]['score'].mean()
            delta_ari_with_scribble = ari_with_scribble - ari_without_scribble
            df_out.loc[sample, 'ari_base_without_hist_' + m] = ari_without_scribble
            df_out.loc[sample, 'delta_ari_with_scribble_' + m] = delta_ari_with_scribble

df_out.to_csv('./results/summary_delta_ari_all_methods.csv')

df_out = pd.DataFrame(
    index=samples,
    columns=['ari_with_hist_conST', 'ari_without_hist_conST', 'ari_with_hist_SpaGCN',
             'ari_without_hist_SpaGCN','ari_with_scribble_ScribbleDom', 'ari_without_scribble_ScribbleDom']
)

for sample in samples:
    # fill out df_out
    for m in method:
        if m in ['conST', 'SpaGCN']:
            ari_with_hist = result_df[(result_df['method'] == m) & (result_df['data_sample'] == sample) & (result_df['sub_path'] == 'w_hist_hihires')]['score'].median()
            ari_without_hist = result_df[(result_df['method'] == m) & (result_df['data_sample'] == sample) & (result_df['sub_path'] == 'wo_hist')]['score'].median()
            df_out.loc[sample, 'ari_without_hist_' + m] = ari_without_hist
            df_out.loc[sample, 'ari_with_hist_' + m] = ari_with_hist
        elif m == 'ScribbleDom':
            ari_with_scribble = result_df[(result_df['method'] == m) & (result_df['data_sample'] == sample) & (result_df['sub_path'] == 'expert')]['score'].median()
            ari_without_scribble = result_df[(result_df['method'] == m) & (result_df['data_sample'] == sample) & (result_df['sub_path'] == 'mclust')]['score'].median()
            df_out.loc[sample, 'ari_without_scribble_' + m] = ari_without_scribble
            df_out.loc[sample, 'ari_with_scribble_' + m] = ari_with_scribble

df_out.to_csv('./results/summary_ari_all_methods.csv')