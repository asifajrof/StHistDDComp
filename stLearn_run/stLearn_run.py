# import scanpy
import gc
import json
import sys
import os
import scanpy as sc
import pandas as pd
import stlearn as st
import matplotlib.pyplot as plt
from anndata import AnnData
from pathlib import Path
from typing import Union
st.settings.set_figure_params(dpi=180)

def ReadStVisium(
    data_path: str,
    count_file: str = "filtered_feature_bc_matrix.h5",
    image_file: Union[str, Path] = None,
    library_id: str = "STVisium",
    quality: str = "fulres",
    spot_diameter_fullres: float = 100,
) -> AnnData:
    count_path = os.path.join(data_path, count_file)
    adata = sc.read(count_path)

    tissue_file = os.path.join(data_path, "spatial", "tissue_positions_list.csv")
    tissue_pos = pd.read_csv(tissue_file, header=None)
    tissue_pos.columns = ["barcode", "tissue", "row", "col", "imagecol", "imagerow"]

    adata.obsm["spatial"] = tissue_pos[["imagerow", "imagecol"]].to_numpy()

    scale_file = os.path.join(data_path, "spatial", "scalefactors_json.json")
    with open(scale_file) as json_file:
        scale_factors = json.load(json_file)

    if quality != "fulres":
        tissue_pos["imagerow"] = (
            tissue_pos["imagerow"] * scale_factors[f"tissue_{quality}_scalef"]
        )
        tissue_pos["imagecol"] = (
            tissue_pos["imagecol"] * scale_factors[f"tissue_{quality}_scalef"]
        )

    adata.obs["barcode"] = adata.obs_names
    adata.obs = adata.obs.merge(
        tissue_pos, left_on="barcode", right_on="barcode", how="left"
    )
    adata.obs_names = adata.obs["barcode"]
    adata.obs.drop(columns=["barcode"], inplace=True)

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    image_coor = adata.obs[["imagerow", "imagecol"]].values
    if quality == "fulres":
        # glob on a path string
        data_path_obj = Path(data_path)
        image_candidates = list(data_path_obj.glob("*.tif"))
        img_path = image_candidates[0]
    elif quality in ["hires", "lowres"]:
        img_path = os.path.join(data_path, "spatial", f"tissue_{quality}_image.png")
    img = plt.imread(img_path, 0)

    adata.uns["spatial"][library_id]["images"][quality] = img
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"][
        "tissue_" + quality + "_scalef"
    ] = scale_factors[f"tissue_{quality}_scalef"]
    adata.uns["spatial"][library_id]["scalefactors"][
        "spot_diameter_fullres"
    ] = spot_diameter_fullres

    return adata

if len(sys.argv) < 2:
    config_file_path = "config.json"
elif len(sys.argv) == 2 or len(sys.argv) == 3:
    config_file_path = sys.argv[1]
else:
    print("Usage: python stLearn_run.py [config_file_path] [seed]")
    exit(0)

with open(config_file_path, mode="r", encoding="utf-8") as config_file:
    config = json.load(config_file)
if len(sys.argv) == 3:
    config["seed"] = int(sys.argv[2])

config["result_path"] = f"{config['result_path']}/w_hist" if config["use_hist"] else f"{config['result_path']}/wo_hist"
config["result_path"] = f"{config['result_path']}/results_seed_{config['seed']}"
data_path = config["data_path"]
result_path = config["result_path"]
BASE_PATH = Path(f"{data_path}")
sample_names = config["sample_names"]
for sample in sample_names:
    print(f"Processing {sample} ...")

    TILE_PATH = Path(f"{result_path}/tmp/{sample}_tiles")
    TILE_PATH.mkdir(parents=True, exist_ok=True)

    # data = st.Read10X(BASE_PATH / sample, library_id="diffusion" if config["use_diffusion"] else None,
                    #   simulated=config["use_diffusion"])
    data = ReadStVisium(BASE_PATH / sample, quality="hires")

    n_cluster = config["n_clusters"][f"{sample}"]

    # pre-processing for gene count table
    st.pp.filter_genes(data, min_cells=1)
    st.pp.normalize_total(data)
    st.pp.log1p(data)

    # run PCA for gene expression data
    st.em.run_pca(data, n_comps=15, random_state=config["seed"])

    # pre-processing for spot image
    st.pp.tiling(data, TILE_PATH)

    # this step uses deep learning model to extract high-level features from tile images
    # may need few minutes to be completed
    st.pp.extract_feature(data, seeds=config["seed"])

    # stSME
    st.spatial.SME.SME_normalize(
        data, use_data="raw", weights="physical_distance")
    # data_ = data.copy()
    data.X = data.obsm["raw_SME_normalized"]

    st.pp.scale(data)
    st.em.run_pca(data, n_comps=15, random_state=config["seed"])

    st.tl.clustering.kmeans(data, n_clusters=n_cluster,
                            use_data="X_pca", key_added="X_pca_kmeans", random_state=config["seed"])

    methods_ = "stSME_disk"

    # saving results
    os.makedirs(f"{result_path}/{sample}", exist_ok=True)

    labels_df = data.obs["X_pca_kmeans"].rename("label")
    labels_df.to_csv(f"{result_path}/{sample}/labels.csv")
    # delete tile folder
    os.system(f"rm -rf {TILE_PATH}")
    print(f"saved to {result_path}/{sample}/labels.csv")

    # clear memory
    del data
    del labels_df
    gc.collect()

    print(f"Done {sample} ...")
