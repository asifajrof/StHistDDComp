from pathlib import Path
from typing import Optional, Union
from anndata import AnnData
import numpy as np
from PIL import Image
import pandas as pd
import scanpy
import matplotlib.pyplot as plt
from matplotlib.image import imread
import json
from tqdm import tqdm
import os
import subprocess


def Read10X(
    path: Union[str, Path],
    genome: Optional[str] = None,
    count_file: str = "filtered_feature_bc_matrix.h5",
    library_id: str = None,
    load_images: Optional[bool] = True,
    quality="fulres",
    image_path: Union[str, Path] = None,
) -> AnnData:
    """\
    [Taken from stLearn]

    Read Visium data from 10X (wrap read_visium from scanpy)
    In addition to reading regular 10x output,
    this looks for the `spatial` folder and loads images,
    coordinates and scale factors.
    Based on the `Space Ranger output docs`_.
    .. _Space Ranger output docs: https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview
    Parameters
    ----------
    path
        Path to directory for visium datafiles.
    genome
        Filter expression to genes within this genome.
    count_file
        Which file in the passed directory to use as the count file. Typically would be one of:
        'filtered_feature_bc_matrix.h5' or 'raw_feature_bc_matrix.h5'.
    library_id
        Identifier for the visium library. Can be modified when concatenating multiple adata objects.
    load_images
        Load image or not.
    quality
        Set quality that convert to stlearn to use. Store in anndata.obs['imagecol' & 'imagerow']
    image_path
        Path to image. Only need when loading full resolution image.
    Returns
    -------
    Annotated data matrix, where observations/cells are named by their
    barcode and variables/genes by gene name. Stores the following information:
    :attr:`~anndata.AnnData.X`
        The data matrix is stored
    :attr:`~anndata.AnnData.obs_names`
        Cell names
    :attr:`~anndata.AnnData.var_names`
        Gene names
    :attr:`~anndata.AnnData.var`\\ `['gene_ids']`
        Gene IDs
    :attr:`~anndata.AnnData.var`\\ `['feature_types']`
        Feature types
    :attr:`~anndata.AnnData.uns`\\ `['spatial']`
        Dict of spaceranger output files with 'library_id' as key
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['images']`
        Dict of images (`'fulres'`, `'hires'` and `'lowres'`)
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['scalefactors']`
        Scale factors for the spots
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['metadata']`
        Files metadata: 'chemistry_description', 'software_version'
    :attr:`~anndata.AnnData.obsm`\\ `['spatial']`
        Spatial spot coordinates, usable as `basis` by :func:`~scanpy.pl.embedding`.
    """

    path = Path(path)
    adata = scanpy.read_10x_h5(path / count_file, genome=genome)

    adata.uns["spatial"] = dict()

    from h5py import File

    with File(path / count_file, mode="r") as f:
        attrs = dict(f.attrs)
    if library_id is None:
        library_id = str(attrs.pop("library_ids")[0], "utf-8")

    adata.uns["spatial"][library_id] = dict()

    tissue_positions_file = (
        path / "spatial/tissue_positions.csv"
        if (path / "spatial/tissue_positions.csv").exists()
        else path / "spatial/tissue_positions_list.csv"
    )

    if load_images:
        files = dict(
            tissue_positions_file=tissue_positions_file,
            scalefactors_json_file=path / "spatial/scalefactors_json.json",
            hires_image=path / "spatial/tissue_hires_image.png",
            lowres_image=path / "spatial/tissue_lowres_image.png",
        )

        # check if files exists, continue if images are missing
        for f in files.values():
            if not f.exists():
                if any(x in str(f) for x in ["hires_image", "lowres_image"]):
                    print(
                        f"You seem to be missing an image file.\n"
                        f"Could not find '{f}'."
                    )
                else:
                    raise OSError(f"Could not find '{f}'")

        adata.uns["spatial"][library_id]["images"] = dict()
        for res in ["hires", "lowres"]:
            try:
                adata.uns["spatial"][library_id]["images"][res] = imread(
                    str(files[f"{res}_image"])
                )
            except Exception:
                raise OSError(f"Could not find '{res}_image'")

        # read json scalefactors
        adata.uns["spatial"][library_id]["scalefactors"] = json.loads(
            files["scalefactors_json_file"].read_bytes()
        )

        adata.uns["spatial"][library_id]["metadata"] = {
            k: (str(attrs[k], "utf-8")
                if isinstance(attrs[k], bytes) else attrs[k])
            for k in ("chemistry_description", "software_version")
            if k in attrs
        }

        # read coordinates
        positions = pd.read_csv(files["tissue_positions_file"], header=None)
        positions.columns = [
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_col_in_fullres",
            "pxl_row_in_fullres",
        ]
        positions.index = positions["barcode"]

        adata.obs = adata.obs.join(positions, how="left")

        adata.obsm["spatial"] = (
            adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]]
            .to_numpy()
            .astype(int)
        )
        adata.obs.drop(
            columns=["barcode", "pxl_row_in_fullres", "pxl_col_in_fullres"],
            inplace=True,
        )

        # put image path in uns
        if image_path is not None:
            # get an absolute path
            image_path = str(Path(image_path).resolve())
            adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(
                image_path
            )

    adata.var_names_make_unique()

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    if quality == "fulres":
        image_coor = adata.obsm["spatial"]
        img = plt.imread(image_path, 0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + quality + "_scalef"
        ]
        image_coor = adata.obsm["spatial"] * scale

    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality

    return adata


def tiling(
    adata: AnnData,
    image,
    out_path: Union[Path, str] = "./tiling",
    library_id: str = None,
    crop_size: int = 40,
    target_size: int = 299,
    verbose: bool = False,
    copy: bool = False,
) -> Optional[AnnData]:
    """\
    [Taken from stLearn]

    Tiling H&E images to small tiles based on spot spatial location

    Parameters
    ----------
    adata
        Annotated data matrix.
    out_path
        Path to save spot image tiles
    library_id
        Library id stored in AnnData.
    crop_size
        Size of tiles
    verbose
        Verbose output
    copy
        Return a copy instead of writing to adata.
    target_size
        Input size for convolutional neuron network
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **tile_path** : `adata.obs` field
        Saved path for each spot image tiles
    """

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    # Check the exist of out_path
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    # image = adata.uns["spatial"][library_id]["images"][
    #     adata.uns["spatial"][library_id]["use_quality"]
    # ]
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    img_pillow = Image.fromarray(image)

    if img_pillow.mode == "RGBA":
        img_pillow = img_pillow.convert("RGB")

    tile_names = []

    with tqdm(
        total=len(adata),
        desc="Tiling image",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for imagerow, imagecol in zip(adata.obs["imagerow"], adata.obs["imagecol"]):
            imagerow_down = imagerow - crop_size / 2
            imagerow_up = imagerow + crop_size / 2
            imagecol_left = imagecol - crop_size / 2
            imagecol_right = imagecol + crop_size / 2
            tile = img_pillow.crop(
                (imagecol_left, imagerow_down, imagecol_right, imagerow_up)
            )
            tile.thumbnail((target_size, target_size), Image.LANCZOS)
            # tile.resize((target_size, target_size))
            tile_name = str(imagecol) + "-" + str(imagerow) + \
                "-" + str(crop_size)
            out_tile = Path(out_path) / (tile_name + ".jpeg")
            tile_names.append(str(out_tile))
            if verbose:
                print(
                    "generate tile at location ({}, {})".format(
                        str(imagecol), str(imagerow)
                    )
                )
            tile.save(out_tile, "JPEG")

            pbar.update(1)

    # adata.obs["tile_path"] = tile_names
    # return adata if copy else None


def main(data_path, image_path, output_folder, mae_folder, model_path):
    # read full adata
    full_adata = Read10X(f'{data_path}', load_images=True,
                         quality="fulres", image_path=f'{image_path}')

    # read image
    pil_img = Image.open(f'{image_path}')
    np_img = np.asarray(pil_img)

    # tiling
    os.makedirs(f'{output_folder}/image_tiles', exist_ok=True)
    tiling(adata=full_adata, image=np_img,
           out_path=f'{output_folder}/image_tiles', crop_size=224, copy=False)

    image_tile_folder_path = Path(f'{output_folder}/image_tiles')
    tile_path_list = list(image_tile_folder_path.iterdir())
    tile_np_array = []
    for tile_path in tile_path_list:
        pil_img = Image.open(tile_path)
        np_img = np.asarray(pil_img)
        tile_np_array.append(np_img)

    tile_np_array = np.array(tile_np_array)
    np.save(f'{output_folder}/image_tiles.npy', tile_np_array)
    subprocess.run(f'rm -rf {output_folder}/image_tiles', shell=True)

    # extract feature
    prev_dir = os.getcwd()
    os.chdir(f'{mae_folder}')
    output_path = f'{output_folder}/extracted_feature.npy'
    patches_path = f'{output_folder}/image_tiles.npy'

    subprocess.run(
        f'python run_mae_extract_feature.py \"{patches_path}\" \"{output_path}\" \"{model_path}\"', shell=True)

    subprocess.run(
        f'rm -rf {output_folder}/image_tiles.npy', shell=True)

    os.chdir(prev_dir)


if __name__ == "__main__":
    print(f'mae_run.py')
