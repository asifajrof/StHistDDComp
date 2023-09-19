import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot(
        run_count,
        data_samples,
        ari_type,
        label_sub_dirs,
        ari_save_base_path,
        plot_base_path
):
    os.makedirs(plot_base_path, exist_ok=True)
    run_no_list = np.arange(1, run_count+1)

    for data_sample in data_samples:
        print(f"Processing: {data_sample}")
        fig, ax = plt.subplots(figsize=(15, 10))
        box_plot_df = pd.DataFrame()
        for label_sub_dir in label_sub_dirs:
            ari_df = pd.read_csv(
                os.path.join(
                    ari_save_base_path,
                    label_sub_dir,
                    f"{data_sample}_ari.csv"
                )
            )
            ax.plot(run_no_list, ari_df[ari_type],
                    marker='o', linestyle='-', label=label_sub_dir)

            ari_df_hist = pd.DataFrame()
            ari_df_hist[ari_type] = ari_df[ari_type]
            ari_df_hist["hist"] = label_sub_dir
            box_plot_df = pd.concat([box_plot_df, ari_df_hist], axis=0)

        ax.set_xlabel('run no')
        ax.set_ylabel('ARI')
        ax.set_title(f'{data_sample}')
        ax.grid(True)
        ax.legend(loc='upper right')
        # ax.set_xlim(0, run_count+1)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, run_count+1, 1))
        ax.set_yticks(np.arange(0, 1, 0.05))
        fig.patch.set_facecolor('white')
        # plt.show()
        fig.savefig(
            os.path.join(
                plot_base_path,
                f"{data_sample}_lineplot.png"
            )
        )
        plt.close()

        sns.set_theme(style="ticks")
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.boxplot(y=box_plot_df[ari_type],
                    x=box_plot_df["hist"], palette="vlag", width=0.2)
        plt.ylim(0, 1)
        ax.grid(True)
        plt.xlabel("Histology")
        plt.ylabel("ARI")
        plt.title(f"{data_sample}")
        # plt.savefig(f"{data_sample}_boxplot.png")
        plt.savefig(
            os.path.join(
                plot_base_path,
                f"{data_sample}_boxplot.png"
            )
        )
        plt.close()
