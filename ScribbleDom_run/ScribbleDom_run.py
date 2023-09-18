import json
import os
import sys
import subprocess

seed = int(sys.argv[1])
os.chdir("ScribbleDom")

config_files_hbc = ["configs/hbc_b1s1/hbc_b1s1_config_expert.json","configs/hbc_b1s1/hbc_b1s1_config_mclust.json"]
config_files_dlpfc = ["configs/human_dlpfc/dlpfc_config_expert.json","configs/human_dlpfc/dlpfc_config_mclust.json"]

def run_scribbledom(config_files_both):
    for config_file_path in config_files_both:
        with open(config_file_path, mode="r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        config["seed_options"] = [seed]
        config["final_output_folder"] = f"../results/{seed}"
        json_object = json.dumps(config,indent=4)

        with open(config_file_path, "w") as outfile:
            outfile.write(json_object)

    os.system("chmod +x run_other_visium.sh")
    os.system(f"./run_other_visium.sh {config_files_both[0]} {config_files_both[1]}")

run_scribbledom(config_files_dlpfc)
run_scribbledom(config_files_hbc)