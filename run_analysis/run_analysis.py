import wilcox
import plot
import ari
import json

config_path = "config.json"
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

annotations_base_path = config["annotations_base_path"]
label_base_path = config["label_base_path"]
label_sub_dirs = config["label_sub_dirs"]
seeds = config["seeds"]
data_samples = config["data_samples"]
ari_save_base_path = config["ari_save_base_path"]
ari_type = config["ari_type"]
plot_base_path = config["plot_base_path"]

print("Calculating ARI\n")
ari.main(
    annotations_base_path=annotations_base_path,
    label_base_path=label_base_path,
    label_sub_dirs=label_sub_dirs,
    seeds=seeds,
    data_samples=data_samples,
    ari_save_base_path=ari_save_base_path
)

print("\nPlotting\n")
plot.plot(
    run_count=len(seeds),
    data_samples=data_samples,
    ari_type=ari_type,
    label_sub_dirs=label_sub_dirs,
    ari_save_base_path=ari_save_base_path,
    plot_base_path=plot_base_path
)

if len(label_sub_dirs) == 2:
    print("\nWilcox analysis\n")
    wilcox.wilcox(
        data_samples=data_samples,
        ari_type=ari_type,
        label_sub_dirs=label_sub_dirs,
        ari_save_base_path=ari_save_base_path
    )
