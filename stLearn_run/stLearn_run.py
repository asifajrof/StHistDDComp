# import scanpy
import gc
import json
import sys
import os
import stlearn as st
from pathlib import Path
st.settings.set_figure_params(dpi=180)

if len(sys.argv) < 2:
    config_file_path = "config.json"
elif len(sys.argv) == 2 or len(sys.argv) == 3:
    config_file_path = sys.argv[1]
else:
    print("Usage: python stlearn-run.py [config_file_path] [seed]")
    exit(0)

with open(config_file_path, mode="r", encoding="utf-8") as config_file:
    config = json.load(config_file)
if len(sys.argv) == 3:
    config["seed"] = int(sys.argv[2])
config["result_path"] = f"{config['result_path']}/seed_{config['seed']}"
data_path = config["data_path"]
result_path = config["result_path"]
BASE_PATH = Path(f"{data_path}")
sample_names = config["sample_names"]
for sample in sample_names:
    print(f"Processing {sample} ...")
    
    TILE_PATH = Path(f"{result_path}/tmp/{sample}_tiles")
    TILE_PATH.mkdir(parents=True, exist_ok=True)

    data = st.Read10X(BASE_PATH / sample, library_id="diffusion" if config["use_diffusion"] else None,
                        simulated=config["use_diffusion"])

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
    labels_df.to_csv(f"{result_path}/{sample}/result.csv")
    # delete tile folder
    os.system(f"rm -rf {TILE_PATH}")
    print(f"saved to {result_path}/{sample}/result.csv")

    # clear memory
    del data
    del labels_df
    gc.collect()

    print(f"Done {sample} ...")
