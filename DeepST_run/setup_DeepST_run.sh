pip install pip --upgrade
git clone https://github.com/asifajrof/DeepST.git
cd DeepST
pip install -r requirements.txt
# cuda
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
# cpu
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
cd ../
pip install Pillow==9.5.0
