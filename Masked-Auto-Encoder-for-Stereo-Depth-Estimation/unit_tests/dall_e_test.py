#from src.utils.builder import build_trainer, build_config
from src.utils.utils import *
from src.utils.builder import *
from utils.fileio import *
from arguments import args
from time import time
from PIL import Image
import numpy as np
import torchvision.transforms as transforms 


# --------------- configure initialisers -----------------
setup_imports()
parser= args.get_parser()
opts, unknown = parser.parse_known_args()

config = build_config(opts)

# setup aws env variables:
s3_client = FileIOClient(config)

# check correct config loaded;
print(config.model_config.name)
USE_DATASET=True

# --------------- import the model ----------------
from src.losses.image_reconstruction import VanillaPerceptualLoss
from src.losses.dall_e.dvae import Dalle_VAE

# --------------- Initialise DALLE dVAE ----------------
# loads the full dall_e model; encoder + decoder
dall_e = Dalle_VAE(config)
# select the encoder only for the feature extractor
feat_extractor = dall_e.encoder


for keys in feat_extractor.state_dict().keys():
    print(keys)

print(feat_extractor)
# dall_e is split into groups i.e. group_1, group_2 ... group_4.
block1 = feat_extractor.blocks.group_1
print('---------------one block-----------------')
print(block1)
# ---------------- create random input data or initiate actual data ----------------

single_image_pred = torch.randn((8,3,224,224))
single_image_gt = torch.randn((8,3,224,224))

# Test out perceptual loss;
perc_loss = VanillaPerceptualLoss(config)

output = perc_loss(single_image_pred, single_image_gt)
# check if output is as expected
for k, v in output.items():
    print(k, v)
