from scipy.stats import wilcoxon
import pandas as pd
import os


def wilcox(
        data_samples,
        ari_type,
        label_sub_dirs,
        ari_save_base_path
):
    for data_sample in data_samples:
        print(f"Processing: {data_sample}")
        ari_dfs = []
        for label_sub_dir in label_sub_dirs:
            ari_df = pd.read_csv(
                os.path.join(
                    ari_save_base_path,
                    label_sub_dir,
                    f"{data_sample}_ari.csv"
                )
            )
            ari_dfs.append(ari_df)
        stat, p = wilcoxon(ari_dfs[0][ari_type], ari_dfs[1][ari_type])
        print(f'stat={stat}, p={p}')
        if p > 0.05:
            print('Probably the same distribution')
        else:
            print('Probably different distributions')

        print()
