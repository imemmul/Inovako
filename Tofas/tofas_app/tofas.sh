#!/bin/bash


echo '2000' | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
ulimit -n 8192
source /home/inovako/anaconda3/etc/profile.d/conda.sh
conda activate TensorRT

python /home/inovako/Desktop/Inovako/Inovako/Tofas/tofas_app/gui_start.py
