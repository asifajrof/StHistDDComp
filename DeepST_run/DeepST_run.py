import sys
sys.path.append("DeepST/deepst")

import subprocess
import json
from DeepST import run
import gc
import os

# from DeepST.deepst.DeepST import run
# from pathlib import Path

if len(sys.argv) < 2:
    config_file_path = "config.json"
elif len(sys.argv) == 2 or len(sys.argv) == 3:
    config_file_path = sys.argv[1]
else:
    print("Usage: python DeepST_run.py [config_file_path] [seed]")
    exit(0)

with open(config_file_path, mode="r", encoding="utf-8") as config_file:
    config = json.load(config_file)

if len(sys.argv) == 3:
    config["seed"] = sys.argv[2]

pca_n_comps = config["pca_n_comps"]
distType = config["distType"]
pre_epochs = config["pre_epochs"]
adjacent_weight = config["adjacent_weight"]
platform = config["platform"]
k = config["k"]
weights = config["weights"]
Conv_type = config["Conv_type"]
pretrain = config["pretrain"]
dim_reduction = config["dim_reduction"]
priori = config["priori"]
linear_encoder_hidden = config["linear_encoder_hidden"]
conv_hidden = config["conv_hidden"]

data_path = os.path.abspath(config["data_path"])
data_names = config["sample_names"]
save_path = os.path.abspath(config["result_path"])

seed = config["seed"]

save_path = os.path.join(save_path, f"seed_{seed}")

for data_name in data_names:
    print(f"processing {data_name} ...")
    save_path = os.path.join(save_path, data_name)
    n_domains = config["n_clusters"][data_name]
    deepen = run(save_path=save_path,
                 platform=platform,
                 pca_n_comps=pca_n_comps,
                 pre_epochs=pre_epochs,  # According to your own hardware, choose the number of training
                 epochs=1000,  # According to your own hardware, choose the number of training
                 Conv_type=Conv_type,  # you can choose GNN types.
                 linear_encoder_hidden=linear_encoder_hidden,
                 conv_hidden=conv_hidden,
                 seed=seed
                 )

    print(f"loading data")
    adata = deepen._get_adata(
        data_path, data_name, count_file=f"filtered_feature_bc_matrix.h5", verbose=False)
    print(f"augmenting data")
    adata = deepen._get_augment(
        adata, adjacent_weight=adjacent_weight, weights=weights, neighbour_k=4,)

    print(f"creating graph")
    graph_dict = deepen._get_graph(
        adata.obsm["spatial"], distType=distType, k=k)

    print(f"fitting data")
    adata = deepen._fit(adata, graph_dict, pretrain=pretrain,
                        dim_reduction=dim_reduction)

    # adata.write(os.path.join(f"{save_path}/{data_name}",
    #             f"{data_name}_fit.h5ad"), compression="gzip")

    print(f"getting cluster data")
    # without using prior knowledge, setting priori = False.
    adata = deepen._get_cluster_data(
        adata, n_domains=n_domains, priori=priori)

    print(f"saving")
    # adata.write_h5ad(f"{save_path}/{data_name}_result.h5ad")
    os.makedirs(save_path, exist_ok=True)
    labels_df = adata.obs[["DeepST_domain", "DeepST_refine_domain"]]
    labels_df.rename(
        columns={"DeepST_domain": "label", "DeepST_refine_domain": "refined_label"}, inplace=True)
    labels_df.to_csv(os.path.join(save_path, "labels.csv"), index=True)

    print(f"cleaning up")
    # delete tiles
    subprocess.run(
        f"rm -rf {save_path}/Image_crop", shell=True)

    # delete objects
    del adata
    del deepen
    del graph_dict

    gc.collect()

    print(f"Done {data_name}.")
