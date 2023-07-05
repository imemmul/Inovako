import os
# TODO this file is just for the categorization

CATEG_DET = ['DET', 'NO_DET']

out_dir = "/home/emir/Desktop/dev/Inovako/Inovako/Tofas/tofas_app/output/"

def categorize_create_folder(out_dir, cams_name, exposures):
    paths = []
    selected_path = ''
    if os.path.exists(out_dir):
        count = len(os.listdir(out_dir))
        run_path = f"run_{count+1}"
        selected_path = os.path.join(out_dir, run_path)
        os.makedirs(selected_path)
    else:
        print(f"path doesn't exist")
    
    for cam in cams_name:
        paths.append(os.path.join(selected_path, cam))
    for path in paths:
        print(f"creating path: {path}")
        if not os.path.exists(path):
            os.makedirs(path)
        for exp in exposures:
            exp = str(exp)
            if not os.path.exists(path):
                os.makedirs(os.path.join(path, exp))
            for cat in CATEG_DET:
                os.makedirs(os.path.join(os.path.join(path, exp), cat))
