import os
OUTPUT_PATH = "/home/emir/Desktop/dev/Inovako/Inovako/Tofas/tofas_app/output/"

for dir in os.listdir(OUTPUT_PATH):
    os.remove(os.path.join(OUTPUT_PATH, dir))