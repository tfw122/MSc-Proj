import os
import argparse
from tqdm import tqdm 
import shutil

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=" /media/neuralmaster/9d5af100-a900-4e89-bab1-43c8b5025daf/neuromaster/MainImportant/MRES/tiaW/Masked-Auto-Encoder-for-Stereo-Depth-Estimation/data/tiawarner", help="the input parent directory")

    args = parser.parse_args()
    return args 


args = get_args()
root_dir = args.input_dir
sub_folders = os.listdir(root_dir)
print("total sub folders where weights are saved: {}".format(len(sub_folders)))

for i in tqdm(range(len(sub_folders))):
    out = "{}/{}/train_outputs/".format(root_dir, sub_folders[i])
    files = os.listdir(out)
    if len(files)==0:
        # if empty; delete folder
        shutil.rmtree("{}/{}".format(root_dir, sub_folders[i]))

    else:
        pass
