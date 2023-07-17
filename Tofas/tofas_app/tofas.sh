#!/bin/bash


echo '2000' | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
ulimit -n 8192
# source /home/inovako/anaconda3/etc/profile.d/conda.sh
source /home/emir/miniconda3/etc/profile.d/conda.sh
# conda activate TensorRT
conda activate mlptorch

# python /home/inovako/Desktop/Inovako/Inovako/Tofas/tofas_app/app.py
python ./app.py