# fmt: off
import sys
sys.path.append("conST")
# fmt: on
import subprocess
import MAE_run
import json
import torch
import argparse
import random
import numpy as np
import pandas as pd
from src.graph_func import graph_construction
from src.utils_func import mk_dir, adata_preprocess, load_ST_file, res_search_fixed_clus, plot_clustering
from src.training import conST_training
import anndata
import matplotlib.pyplot as plt
import scanpy as sc
import os
import warnings


warnings.filterwarnings("ignore")

LOGGING = True


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def refine(sample_id, pred, dis, shape="hexagon"):
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        print(
            "Shape not recongized, shape=\"hexagon\" for Visium data, \"square\" for ST data.")
    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :].sort_values(ascending=False)
        nbs = dis_tmp[0:num_nbs+1]
        nbs_pred = pred.loc[nbs.index, "pred"]
        self_pred = pred.loc[index, "pred"]
        v_c = nbs_pred.value_counts()
        if (v_c.loc[self_pred] < num_nbs/2) and (np.max(v_c) > num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred


if len(sys.argv) < 2:
    config_file_path = "config.json"
elif len(sys.argv) == 2 or len(sys.argv) == 3:
    config_file_path = sys.argv[1]
else:
    print("Usage: python conST_run.py [config_file_path] [seed]")
    exit(0)

with open(config_file_path, mode="r", encoding="utf-8") as config_file:
    config = json.load(config_file)

if len(sys.argv) == 3:
    config["seed"] = sys.argv[2]

config["result_path"] = f"{config['result_path']}/w_hist" if config["use_hist"] else f"{config['result_path']}/wo_hist"
config["result_path"] = f"{config['result_path']}/results_seed_{config['seed']}"

# ______________ params ______________
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=10,
                    help="parameter k in spatial graph")
parser.add_argument("--knn_distanceType", type=str, default="euclidean",
                    help="graph distance type: euclidean/cosine/correlation")
parser.add_argument("--epochs", type=int, default=200,
                    help="Number of epochs to train.")
parser.add_argument("--cell_feat_dim", type=int,
                    default=290, help="Dim of PCA")
parser.add_argument("--morph_feat_dim", type=int, default=768,
                    help="Dim of Morphological feat for PCA")
parser.add_argument("--feat_hidden1", type=int, default=100,
                    help="Dim of DNN hidden 1-layer.")
parser.add_argument("--feat_hidden2", type=int, default=20,
                    help="Dim of DNN hidden 2-layer.")
parser.add_argument("--gcn_hidden1", type=int, default=32,
                    help="Dim of GCN hidden 1-layer.")
parser.add_argument("--gcn_hidden2", type=int, default=8,
                    help="Dim of GCN hidden 2-layer.")
parser.add_argument("--p_drop", type=float,
                    default=0.2, help="Dropout rate.")
parser.add_argument("--use_img", type=bool, default=False,
                    help="Use histology images.")
parser.add_argument("--img_w", type=float, default=0.1,
                    help="Weight of image features.")
parser.add_argument("--use_pretrained", type=bool,
                    default=True, help="Use pretrained weights.")
parser.add_argument("--using_mask", type=bool, default=False,
                    help="Using mask for multi-dataset.")
parser.add_argument("--feat_w", type=float, default=10,
                    help="Weight of DNN loss.")
parser.add_argument("--gcn_w", type=float, default=0.1,
                    help="Weight of GCN loss.")
parser.add_argument("--dec_kl_w", type=float, default=10,
                    help="Weight of DEC loss.")
parser.add_argument("--gcn_lr", type=float, default=0.01,
                    help="Initial GNN learning rate.")
parser.add_argument("--gcn_decay", type=float,
                    default=0.01, help="Initial decay rate.")
parser.add_argument("--dec_cluster_n", type=int,
                    default=10, help="DEC cluster number.")
parser.add_argument("--dec_interval", type=int, default=20,
                    help="DEC interval nnumber.")
parser.add_argument("--dec_tol", type=float, default=0.00, help="DEC tol.")

parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--beta", type=float, default=100,
                    help="beta value for l2c")
parser.add_argument("--cont_l2l", type=float, default=0.3,
                    help="Weight of local contrastive learning loss.")
parser.add_argument("--cont_l2c", type=float, default=0.1,
                    help="Weight of context contrastive learning loss.")
parser.add_argument("--cont_l2g", type=float, default=0.1,
                    help="Weight of global contrastive learning loss.")

parser.add_argument("--edge_drop_p1", type=float, default=0.1,
                    help="drop rate of adjacent matrix of the first view")
parser.add_argument("--edge_drop_p2", type=float, default=0.1,
                    help="drop rate of adjacent matrix of the second view")
parser.add_argument("--node_drop_p1", type=float, default=0.2,
                    help="drop rate of node features of the first view")
parser.add_argument("--node_drop_p2", type=float, default=0.3,
                    help="drop rate of node features of the second view")

# ______________ Eval clustering Setting ______________
parser.add_argument("--eval_resolution", type=int,
                    default=1, help="Eval cluster number.")
parser.add_argument("--eval_graph_n", type=int,
                    default=20, help="Eval graph kN tol.")

params = parser.parse_args(
    args=[
        "--k", f"{config['k_knn']}",
        "--knn_distanceType", f"{config['knn_distanceType']}",
        "--epochs", f"{config['epochs']}",
        "--use_img", f"{config['use_hist']}",
        "--seed", f"{config['seed']}",
    ])

params.use_pretrained = config["use_pretrained"]
params.use_img = config["use_hist"]

np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if LOGGING:
    print("Using device: " + device)
params.device = device

# ______________ Set Paths ______________
data_folder = config["data_path"]
input_folder = config["input_path"]
save_folder = config["result_path"]

# ______________ Save Params ______________
params_dict = vars(params)
params_df = pd.DataFrame.from_dict(params_dict, orient="index")
os.makedirs(save_folder, exist_ok=True)
params_df.to_csv(os.path.join(save_folder, "params.csv"), header=False)

for sample_name in config["sample_names"]:
    if LOGGING:
        print(f"processing {sample_name}")

    # ______________ Load Data ______________
    if LOGGING:
        print("Loading data")
    path = os.path.join(data_folder, sample_name)
    adata_h5 = load_ST_file(
        path, count_file="filtered_feature_bc_matrix.h5")

    # ______________ Preprocess Data ______________
    if LOGGING:
        print("Preprocessing data")
    os.makedirs(os.path.join(input_folder, sample_name), exist_ok=True)
    adatax_path = os.path.join(input_folder, sample_name, "adatax.npy")
    graphdict_path = os.path.join(
        input_folder, sample_name, "graphdict.npy")

    adata_X = adata_preprocess(
        adata_h5, min_cells=5, pca_n_comps=params.cell_feat_dim)
    graph_dict = graph_construction(
        adata_h5.obsm["spatial"], adata_h5.shape[0], params)

    # if not os.path.exists(adatax_path) or not os.path.exists(graphdict_path):
    #     adata_X = adata_preprocess(
    #         adata_h5, min_cells=5, pca_n_comps=params.cell_feat_dim)
    #     graph_dict = graph_construction(
    #         adata_h5.obsm["spatial"], adata_h5.shape[0], params)

    #     np.save(adatax_path, adata_X)
    #     np.save(graphdict_path, graph_dict, allow_pickle = True)

    # else:
    #     adata_X = np.load(adatax_path)
    #     graph_dict = np.load(graphdict_path, allow_pickle=True).item()

    # ______________ Run MAE ______________
    if params.use_img:
        if not os.path.exists(os.path.join(input_folder, sample_name, "extracted_feature.npy")):
            if LOGGING:
                print("Running MAE")
            MAE_run.main(
                data_path=os.path.join(os.getcwd(), data_folder, sample_name),
                image_path=os.path.join(
                    os.getcwd(), data_folder, sample_name, f"{sample_name}_full_image.tif"),
                output_folder=os.path.join(
                    os.getcwd(), input_folder, sample_name),
                mae_folder=os.path.join(os.getcwd(), "conST", "MAE-pytorch"),
                model_path=os.path.join(os.getcwd(), config["mae_path"])
            )

    # ______________ Run conST ______________
    seed_torch(params.seed)

    params.save_path = mk_dir(os.path.join(save_folder, sample_name))

    # path = os.path.join(data_folder, sample_name)
    params.cell_num = adata_h5.shape[0]

    n_clusters = config["n_clusters"][sample_name]
    if LOGGING:
        print("Running conST")
    if params.use_img:
        img_transformed = np.load(os.path.join(
            input_folder, sample_name, "extracted_feature.npy"))
        img_transformed = (img_transformed - img_transformed.mean()) / \
            img_transformed.std() * adata_X.std() + adata_X.mean()
        conST_net = conST_training(
            adata_X, graph_dict, params, n_clusters, img_transformed)
    else:
        conST_net = conST_training(adata_X, graph_dict, params, n_clusters)

    if params.use_pretrained:
        if LOGGING:
            print("using pretrained")
        conST_net.load_model(config["pretrained_path"][sample_name])
    else:
        if LOGGING:
            print("model training")
        conST_net.pretraining()
        conST_net.major_training()

    if LOGGING:
        print("get embedding")
    conST_embedding = conST_net.get_embedding()

    # ______________ Run Clustering ______________
    if LOGGING:
        print("run clustering")
    adata_conST = anndata.AnnData(conST_embedding)
    adata_conST.uns["spatial"] = adata_h5.uns["spatial"]
    adata_conST.obsm["spatial"] = adata_h5.obsm["spatial"]

    sc.pp.neighbors(adata_conST, n_neighbors=params.eval_graph_n)

    eval_resolution = res_search_fixed_clus(adata_conST, n_clusters)
    print(eval_resolution)
    cluster_key = "conST_leiden"
    sc.tl.leiden(adata_conST, key_added=cluster_key,
                 resolution=eval_resolution)

    # ______________ Refine Clustering ______________
    if LOGGING:
        print("refine clustering")
    index = np.arange(start=0, stop=adata_X.shape[0]).tolist()
    index = [str(x) for x in index]

    dis = graph_dict["adj_norm"].to_dense().numpy(
    ) + np.eye(graph_dict["adj_norm"].shape[0])
    refine_cluster = refine(sample_id=index,
                            pred=adata_conST.obs["leiden"].tolist(), dis=dis, shape="square")
    adata_conST.obs["refine"] = refine_cluster

    # ______________ Save Results ______________
    if LOGGING:
        print("save results")
    adata_conST.obs["imagecol"] = adata_conST.obsm["spatial"][:, 0]
    adata_conST.obs["imagerow"] = adata_conST.obsm["spatial"][:, 1]
    # adata_conST.write(f"{params.save_path}/conST_result.h5ad")
    os.makedirs(params.save_path, exist_ok=True)
    labels_df = adata_conST.obs[["conST_leiden", "refine"]]
    labels_df.rename(
        columns={"conST_leiden": "label", "refine": "refined_label"}, inplace=True)
    labels_df.to_csv(os.path.join(params.save_path, "labels.csv"), index=True)

    # ______________ Clean Up ______________
    if LOGGING:
        print("clean up")
    del adata_h5
    del adata_X
    del graph_dict
    del conST_net
    del conST_embedding
    del adata_conST
    if params.use_img:
        subprocess.run(
            f"rm -rf {input_folder}/{sample_name}/extracted_feature.npy", shell=True)
    if LOGGING:
        print(f"done {sample_name}")
    print("__________________________")
