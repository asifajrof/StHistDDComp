#!/bin/bash
git clone https://github.com/1alnoman/ScribbleDom.git
conda env create -f environment.yml
conda activate scribble_dom
python run_20.py