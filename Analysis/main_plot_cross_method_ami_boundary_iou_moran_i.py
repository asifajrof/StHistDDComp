import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_samples = ["151507","151508","151509","151510","151669","151670","151671","151672","151673","151674","151675","151676","hbc","bcdc","melanoma"]
    for sample in data_samples:
        df_iou_of_boundary = pd.read_csv(f"results/iou_confusion_matrices_on_boundary/{sample}/iou_confusion_matrix.csv", index_col=0)
        df_ami = pd.read_csv(f"results/ami_confusion_matrices/{sample}/ami_confusion_matrix.csv", index_col=0)
        df_moran_i_of_boundary = pd.read_csv(f"results/morans_i_on_boundary/{sample}/morans_i_boundary_scores.csv", index_col=0)

        all_indices_of_interest = []
        replace_map_index_names = {}
        for idx, index_name in enumerate(df_iou_of_boundary.index):
            if index_name.lower().endswith("_large"):
                continue
            else:
                curr_index = index_name
                curr_index = curr_index.replace("_hihires", "")

                all_indices_of_interest.append(index_name)
                replace_map_index_names[index_name] = curr_index

        df_iou_filtered = df_iou_of_boundary.loc[all_indices_of_interest, all_indices_of_interest]
        df_iou_filtered = df_iou_filtered.rename(index=replace_map_index_names, columns=replace_map_index_names)
        df_ami_filtered = df_ami.loc[all_indices_of_interest, all_indices_of_interest]
        df_ami_filtered = df_ami_filtered.rename(index=replace_map_index_names, columns=replace_map_index_names)
        df_moran_i_of_boundary = df_moran_i_of_boundary.loc[all_indices_of_interest, :]
        df_moran_i_of_boundary = df_moran_i_of_boundary.rename(index=replace_map_index_names)

        sns.heatmap(df_iou_filtered, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1)

        plt.title('IoU Overlap of Boundary Spots', fontsize=16)
        plt.xticks(rotation=90, ha='right') # Rotate x-axis labels for readability
        plt.yticks(rotation=0)
        fig = plt.gcf()

        out_dir = "results/plots/conf_matrix_iou_boundary/"
        import os
        os.makedirs(out_dir, exist_ok=True)
        outpath = f"{out_dir}/{sample}_iou_confusion_matrix_boundary_filtered_Set1.pdf"
        if outpath.endswith('.eps'):
            fig.savefig(outpath, format='eps', dpi=300, bbox_inches='tight', pad_inches=0.1)
        else:
            fig.savefig(outpath, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        sns.heatmap(df_ami_filtered, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1)

        plt.title('AMI of labels', fontsize=16)
        plt.xticks(rotation=90, ha='right') # Rotate x-axis labels for readability
        plt.yticks(rotation=0)
        fig = plt.gcf()
        out_dir = "results/plots/conf_matrix_ami/"
        import os
        os.makedirs(out_dir, exist_ok=True)
        outpath = f"{out_dir}/{sample}_ami_confusion_matrix_filtered_Set1.pdf"
        if outpath.endswith('.eps'):
            fig.savefig(outpath, format='eps', dpi=300, bbox_inches='tight', pad_inches=0.1)
        else:
            fig.savefig(outpath, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        sns.heatmap(df_moran_i_of_boundary, annot=True, fmt=".3f", cmap="Reds", square=True)
        plt.title("Moran's I of Boundary Spots", fontsize=16)
        plt.yticks(rotation=0)  # Ensure y-axis labels are horizontal
        fig = plt.gcf()
        out_dir = "results/plots/morans_i_boundary/"
        import os
        os.makedirs(out_dir, exist_ok=True)
        outpath = f"{out_dir}/{sample}_morans_i_boundary_filtered_Set1.pdf"
        if outpath.endswith('.eps'):
            fig.savefig(outpath, format='eps', dpi=300, bbox_inches='tight', pad_inches=0.1)
        else:
            fig.savefig(outpath, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)