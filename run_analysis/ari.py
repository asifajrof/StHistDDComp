import pandas as pd
import os
from sklearn.metrics import adjusted_rand_score


def get_ari(
        annotations_base_path,
        label_base_path,
        label_sub_dirs,
        seeds,
        data_samples
):
    for label_sub_dir in label_sub_dirs:
        print(f"Processing: {label_sub_dir}")
        for seed in seeds:
            print(f"seed: {seed}")
            ari_df = pd.DataFrame(
                columns=['sample_name', 'seed', 'pred_ari', 'refined_ari'])
            for data_sample in data_samples:
                print(f"data sample: {data_sample}")
                manual_annotations_path = os.path.join(
                    annotations_base_path,
                    f"{data_sample}_annotations.csv"
                )
                if not os.path.exists(manual_annotations_path):
                    print(f"Not exist: {manual_annotations_path}")
                    continue
                manual_annotations_df = pd.read_csv(
                    manual_annotations_path, index_col=0)

                labels_path = os.path.join(
                    label_base_path,
                    label_sub_dir,
                    f"results_seed_{seed}",
                    f"{data_sample}",
                    "labels.csv"
                )
                if not os.path.exists(labels_path):
                    print(f"Not exist: {labels_path}")
                    continue
                print(f"Processing: {labels_path}")
                labels_df = pd.read_csv(labels_path, index_col=0)

                try:
                    labels_for_ari = labels_df[[
                        'label', 'refined_label'
                    ]]
                except KeyError:
                    labels_for_ari = labels_df[['label']]
                    labels_for_ari['refined_label'] = labels_for_ari['label']

                labels_for_ari['manual_annot'] = manual_annotations_df['label'].values
                labels_for_ari['manual_annot'] = labels_for_ari['manual_annot'].fillna(
                    "NA").astype('category')
                filtered_labels_for_ari = labels_for_ari.loc[labels_for_ari["manual_annot"] != "NA"]

                ari = adjusted_rand_score(
                    labels_true=filtered_labels_for_ari['manual_annot'], labels_pred=filtered_labels_for_ari['label'])
                refined_ari = adjusted_rand_score(
                    labels_true=filtered_labels_for_ari['manual_annot'], labels_pred=filtered_labels_for_ari['refined_label'])
                ari_df = pd.concat([ari_df, pd.DataFrame({'sample_name': [data_sample], 'seed': [seed], 'pred_ari': [ari],
                                    'refined_ari': [refined_ari]})], ignore_index=True)

            ari_df.to_csv(
                os.path.join(
                    label_base_path,
                    label_sub_dir,
                    f"results_seed_{seed}",
                    "ari.csv"
                ),
                index=False
            )


def ari_by_data_sample(
        ari_save_base_path,
        label_base_path,
        label_sub_dirs,
        seeds,
        data_samples
):
    for label_sub_dir in label_sub_dirs:
        print(f"Processing: {label_sub_dir}")
        os.makedirs(
            os.path.join(
                ari_save_base_path,
                label_sub_dir
            ),
            exist_ok=True
        )
        for data_sample in data_samples:
            print(f"data sample: {data_sample}")
            main_df = pd.DataFrame()
            for seed in seeds:
                csv_path = os.path.join(
                    label_base_path,
                    label_sub_dir,
                    f"results_seed_{seed}",
                    "ari.csv"
                )
                if not os.path.exists(csv_path):
                    print(f"Not exist: {csv_path}")
                    continue
                print(f"Processing: {csv_path}")
                ari_df = pd.read_csv(csv_path, index_col=0)
                main_df = pd.concat(
                    [main_df, ari_df.loc[[data_sample]]], axis=0)
            main_df["seed"] = seeds
            main_df["sample_name"] = main_df.index
            main_df = main_df[["sample_name",
                               "seed", "pred_ari", "refined_ari"]]
            main_df.to_csv(
                os.path.join(
                    ari_save_base_path,
                    label_sub_dir,
                    f"{data_sample}_ari.csv"
                )
            )


def main(
        annotations_base_path,
        label_base_path,
        label_sub_dirs,
        seeds,
        data_samples,
        ari_save_base_path
):
    get_ari(
        annotations_base_path,
        label_base_path,
        label_sub_dirs,
        seeds,
        data_samples
    )
    ari_by_data_sample(
        ari_save_base_path,
        label_base_path,
        label_sub_dirs,
        seeds,
        data_samples
    )
