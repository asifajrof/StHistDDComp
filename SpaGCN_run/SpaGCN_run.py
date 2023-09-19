import sys
import json
import cv2
from scanpy import read_10x_h5
import os
import pandas as pd
import numpy as np
import scanpy as sc
import SpaGCN as spg
import random
import torch
import warnings
warnings.filterwarnings("ignore")

if len(sys.argv) < 2:
    config_file_path = "config.json"
elif len(sys.argv) == 2 or len(sys.argv) == 3:
    config_file_path = sys.argv[1]
else:
    print("Usage: python SpaGCN_run.py [config_file_path] [seed]")
    exit(0)

with open(config_file_path, mode="r", encoding="utf-8") as config_file:
    config = json.load(config_file)
if len(sys.argv) == 3:
    config["seed"] = int(sys.argv[2])
data_path = config["data_path"]
config["result_path"] = f"{config['result_path']}/w_hist" if config["use_hist"] else f"{config['result_path']}/wo_hist"
result_path = f"{config['result_path']}/results_seed_{config['seed']}"
sample_names = config["sample_names"]

for sample_name in sample_names:
    print(f"Processing {sample_name}")
    # Read original 10x_h5 data and save it to h5ad
    if config["use_diffusion"]:
        adata = sc.read(
            f"{data_path}/{sample_name}/filtered_feature_bc_matrix.h5")
    else:
        adata = read_10x_h5(
            f"{data_path}/{sample_name}/filtered_feature_bc_matrix.h5")
    spatial = pd.read_csv(f"{data_path}/{sample_name}/spatial/tissue_positions_list.csv",
                          sep=",", header=None, na_filter=False, index_col=0)
    adata.obs["x1"] = spatial[1]
    adata.obs["x2"] = spatial[2]
    adata.obs["x3"] = spatial[3]
    adata.obs["x4"] = spatial[4]
    adata.obs["x5"] = spatial[5]
    adata.obs["x_array"] = adata.obs["x2"]
    adata.obs["y_array"] = adata.obs["x3"]
    adata.obs["x_pixel"] = adata.obs["x4"]
    adata.obs["y_pixel"] = adata.obs["x5"]
    # Select captured samples
    adata = adata[adata.obs["x1"] == 1]
    adata.var_names = [i.upper() for i in list(adata.var_names)]
    adata.var["genename"] = adata.var.index.astype("str")

    # Set coordinates
    x_array = adata.obs["x_array"].tolist()
    y_array = adata.obs["y_array"].tolist()
    x_pixel = adata.obs["x_pixel"].tolist()
    y_pixel = adata.obs["y_pixel"].tolist()

    if config["use_hist"]:
        # Read in hitology image
        img = cv2.imread(
            f"{data_path}/{sample_name}/{sample_name}_full_image.tif")

        # Test coordinates on the image
        img_new = img.copy()
        for i in range(len(x_pixel)):
            x = x_pixel[i]
            y = y_pixel[i]
            img_new[int(x-20):int(x+20), int(y-20):int(y+20), :] = 0

        # # not necessary to save the map image
        # cv2.imwrite(f"{result_folder}/{sample_name}_map.jpg", img_new)
    # Calculate adjacent matrix
    ###################################
    # hyperparameters
    s = config["s"]
    b = config["b"]
    ###################################
    if config["use_hist"]:
        adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, x_pixel=x_pixel,
                                       y_pixel=y_pixel, image=img, beta=b, alpha=s, histology=True)
    else:
        # If histlogy image is not available, SpaGCN can calculate the adjacent matrix using the fnction below
        # adj = spg.calculate_adj_matrix(x=x_pixel,y=y_pixel, histology=False)
        adj = spg.calculate_adj_matrix(
            x=x_array, y=y_array, beta=b, alpha=s, histology=False)

    # # not necessary to save the adjacent matrix
    # np.savetxt(f"{data_folder}/{sample_name}/adj.csv", adj, delimiter=",")

    # adata=sc.read(f"{data_folder}/{sample_name}/sample_data.h5ad")
    # adj=np.loadtxt(f"{data_folder}/{sample_name}/adj.csv", delimiter=",")

    adata.var_names_make_unique()
    spg.prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
    spg.prefilter_specialgenes(adata)
    # Normalize and take log for UMI
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    ###################################
    # hyperparameters
    p = config["p"]
    ###################################
    # Find the l value given p
    l = spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

    # If the number of clusters known, we can use the spg.search_res() fnction to search for suitable resolution(optional)
    # For this toy data, we set the number of clusters=7 since this tissue has 7 layers

    ###################################
    # parameters
    n_clusters = config["n_clusters"][sample_name]
    ###################################
    # Set seed
    r_seed = t_seed = n_seed = config["seed"]
    # Seaech for suitable resolution
    res = spg.search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3,
                         lr=config["lr_res"], max_epochs=config["max_epoch_res"], r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)

    clf = spg.SpaGCN()
    clf.set_l(l)
    # Set seed
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    # Run
    clf.train(adata, adj, init_spa=True, init="louvain",
              res=res, tol=5e-3, lr=config["lr_clf"], max_epochs=config["max_epoch_clf"])
    y_pred, prob = clf.predict()
    adata.obs["pred"] = y_pred
    adata.obs["pred"] = adata.obs["pred"].astype("category")
    # Do cluster refinement(optional)
    # shape="hexagon" for Visium data, "square" for ST data.
    adj_2d = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
    refined_pred = spg.refine(sample_id=adata.obs.index.tolist(
    ), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
    adata.obs["refined_pred"] = refined_pred
    adata.obs["refined_pred"] = adata.obs["refined_pred"].astype("category")
    # Save results
    os.makedirs(f"{result_path}/{sample_name}", exist_ok=True)
    labels_df = adata.obs[["pred", "refined_pred"]]
    labels_df.rename(
        columns={"pred": "label", "refined_pred": "refined_label"}, inplace=True)
    labels_df.to_csv(f"{result_path}/{sample_name}/labels.csv", index=True)

    print("Done!")
    print("Results saved in the folder: ", f"{result_path}/{sample_name}")
