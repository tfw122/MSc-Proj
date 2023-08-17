from glob import glob
import os
import zipfile
from tqdm import tqdm

root_dir = "/data/stereo_data/tartan_air_og_full/tartan-air-dataset-og/tartan_air/tartanair_tools/outputs"
output_dir = "/data/stereo_data/tartan_air_extracted"
mode = "Hard" # can unpack "Easy" or "Hard"
items = ["depth_left.zip", "image_left.zip", "image_right.zip"]
extension = ".zip"

locations = os.listdir(root_dir)
print(locations)

for i in tqdm(locations):
    sub_dir = "{}/{}/{}".format(root_dir, i, mode)
    for item in items: #os.listdir(sub_dir):
        #if item.endswith(extension):
        file_name = "{}/{}".format(sub_dir, item) # get full path of files
        zip_ref = zipfile.ZipFile(file_name) # create zipfile object
        zip_ref.extractall(output_dir) # extract file to dir
        zip_ref.close() # close file

print("Extraction Complete!")