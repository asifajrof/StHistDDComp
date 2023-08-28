import os
import sys


def DLPFC(dataset_sample, download_full_image=False):
    if download_full_image:
        os.system(
            f"wget https://spatial-dlpfc.s3.us-east-2.amazonaws.com/images/{dataset_sample}_full_image.tif")
    os.system(
        f"wget -O filtered_feature_bc_matrix.h5 https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/{dataset_sample}_filtered_feature_bc_matrix.h5")
    os.makedirs("spatial", exist_ok=True)
    os.chdir("spatial")
    os.system(
        f"wget https://raw.githubusercontent.com/LieberInstitute/HumanPilot/master/10X/{dataset_sample}/scalefactors_json.json")
    os.system(
        f"wget -O tissue_positions_list.csv https://raw.githubusercontent.com/LieberInstitute/HumanPilot/master/10X/{dataset_sample}/tissue_positions_list.txt")
    os.system(
        f"wget -O tissue_hires_image.png https://spatial-dlpfc.s3.us-east-2.amazonaws.com/images/{dataset_sample}_tissue_hires_image.png")
    os.system(
        f"wget -O tissue_lowres_image.png https://spatial-dlpfc.s3.us-east-2.amazonaws.com/images/{dataset_sample}_tissue_lowres_image.png")
    os.chdir("..")


def HBC(dataset_sample, download_full_image=False):
    if download_full_image:
        os.system(
            f"wget -O {dataset_sample}_full_image.tif https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Breast_Cancer_Block_A_Section_1/V1_Breast_Cancer_Block_A_Section_1_image.tif")
    os.system(f"wget -O filtered_feature_bc_matrix.h5 https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Breast_Cancer_Block_A_Section_1/V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5")
    os.system(f"wget https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Breast_Cancer_Block_A_Section_1/V1_Breast_Cancer_Block_A_Section_1_spatial.tar.gz")
    os.system(f"tar -xf V1_Breast_Cancer_Block_A_Section_1_spatial.tar.gz")
    os.system(f"rm -f V1_Breast_Cancer_Block_A_Section_1_spatial.tar.gz")


def BCDC(dataset_sample, download_full_image=False):
    if download_full_image:
        os.system(
            f"wget -O {dataset_sample}_full_image.tif https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Human_Breast_Cancer/Visium_FFPE_Human_Breast_Cancer_image.tif")
    os.system(f"wget -O filtered_feature_bc_matrix.h5 https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Human_Breast_Cancer/Visium_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5")
    os.system(f"wget https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Human_Breast_Cancer/Visium_FFPE_Human_Breast_Cancer_spatial.tar.gz")
    os.system(f"tar -xf Visium_FFPE_Human_Breast_Cancer_spatial.tar.gz")
    os.system(f"rm -f Visium_FFPE_Human_Breast_Cancer_spatial.tar.gz")


def download_dataset(dataset_name, download_full_image=True, save_path=".", sample_list=[]):
    if len(sample_list) == 0:
        if dataset_name == "DLPFC":
            sample_list = [
                "151507",
                "151508",
                "151509",
                "151510",
                "151669",
                "151670",
                "151671",
                "151672",
                "151673",
                "151674",
                "151675",
                "151676"
            ]
        elif dataset_name == "HBC":
            sample_list = [
                "V1_Breast_Cancer_Block_A_Section_1"
            ]
        elif dataset_name == "BCDC":
            sample_list = [
                "Visium_FFPE_Human_Breast_Cancer"
            ]
    previous_dir = os.getcwd()
    download_path = os.path.abspath(os.path.join(save_path, dataset_name))
    os.makedirs(download_path, exist_ok=True)
    for dataset_sample in sample_list:
        os.chdir(download_path)
        os.makedirs(dataset_sample, exist_ok=True)
        os.chdir(dataset_sample)

        if dataset_name == "DLPFC":
            DLPFC(dataset_sample, download_full_image)
        elif dataset_name == "HBC":
            HBC(dataset_sample, download_full_image)
        elif dataset_name == "BCDC":
            BCDC(dataset_sample, download_full_image)
        else:
            print("Dataset not found!")
    os.chdir(previous_dir)


if __name__ == "__main__":
    # if len(sys.argv) == 2:
    #     dataset_name = sys.argv[1]
    # else:
    #     print("Usage: python download_st_data.py <dataset_name>")
    #     exit(0)

    # download_dataset(dataset_name)
    download_dataset("DLPFC", sample_list=["151673", "151676"])
