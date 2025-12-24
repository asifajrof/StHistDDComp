# histnohist

## Usage

First, install the package dependencies:
```bash
poetry install --no-root
```

Execute each script in order. This will download the data from: (https://drive.google.com/drive/folders/1Dj2f8JlJGBpOxx1yEfNs0WWzrBq3hRKN?usp=sharing) and execute all the analysis.
```bash
python main_find_best_seed.py
python main_calculate_ARI_silh_moranI_gearyC_all_seed.py
python maiin_calculate_delta_ari_and_avg_ari.py
python main_find_ami_on_across_method_for_best_seed.py
python main_iou_confusion_on_boundary.py
python main_morans_i_on_boundary.py
python main_plot_cross_method_ami_boundary_iou_moran_i.py
python main_plot_overlay.py
python main_significance.py
```