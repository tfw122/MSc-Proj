#from src.utils.builder import build_trainer, build_config
from src.utils.utils import *
from src.utils.builder import *
from utils.fileio import *
from arguments import args
from time import time
from PIL import Image
import numpy as np
import torchvision.transforms as transforms 
import torch 
import torch.nn as nn
import timm 
from thop import profile

# --------------- configure initialisers -----------------
setup_imports()
parser= args.get_parser()
opts, unknown = parser.parse_known_args()

config = build_config(opts)

# setup aws env variables:
s3_client = FileIOClient(config)

# check correct config loaded;
print(config.model_config.name)
USE_DATASET=False

from src.models.modules.image_encoder import *
from functools import partial 

# --------------- import the model ----------------
from src.models.model_zoo.mil_classifier_attn_pooling import MILClassifierAttnPooling
from src.models.model_zoo.mil_classifier import MILClassifier
from src.models.model_zoo.language_vision import LanguageVisionMAEModel
from src.models.model_zoo.masked_vision_model import *
from src.models.modules.masked_vision_layers import *
from src.models.model_zoo.vector_quantized_mae import VQMaskedImageAutoEncoder
from src.models.modules.vqgan_modules.taming.models.vqgan import VQModel

# --------------- Initialise the model ----------------
#model = MaskedImageAutoEncoderMSGGAN(config)
img_size = config.dataset_config.preprocess.vision_transforms.params.Resize.size
patch_size = config.model_config.image_encoder.patch_size
in_channels = config.model_config.image_encoder.in_channels
embed_dim = config.model_config.image_encoder.embed_dim
norm_layer_arg= config.model_config.norm_layer_arg
        
if norm_layer_arg=='partial':
    norm_layer = partial(nn.LayerNorm, eps=1e-6)
    print('using partial layer norm')
else:
    norm_layer = nn.LayerNorm

patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)

full_model = VQMaskedImageAutoEncoder(config)
model_config = config.model_config
vq_config = model_config.vector_quantizer
ddconfig = vq_config.ddconfig
lossconfig = vq_config.lossconfig

vqgan = VQModel(ddconfig,
                 lossconfig,
                 vq_config.n_embed,
                 vq_config.embed_dim)

# ---------------- create random input data or initiate actual data ----------------
if USE_DATASET:
    from src.datasets.dataset_zoo.tractable_language_vision.language_vision_builder import TractableLanguageVisionDatasetModule
    from torchvision.transforms import transforms

    #vision_builder = TractableVisionTestDatasetModule(config)
    vision_builder = TractableLanguageVisionDatasetModule(config)
    dataset= vision_builder.data_setup('train')
    # iterate over the dataset
    print('no. # samples: {}'.format(len(dataset)))
    sample = dataset[0]
    train_loader = vision_builder.train_dataloader()

    # ----------------- model inference ----------------
    #masked_sample, itm_samples = sample
    for i, data in enumerate(train_loader):
        if i > 0:
            break
        else:
            masked_sample, itm_samples = data
            output = model(masked_sample)
    

else:
    # create random input tensor:
    n_images=config.dataset_config.max_images
    n_classes=11
    rand_multi_image = torch.randn((8,3,n_images,224,224))
    single_image = torch.randn((8,3,224,224))
    pad_mask = torch.zeros((8,n_images))
    label = torch.zeros((8, n_classes))
    # ----------------- model inference ----------------
    output, _ = vqgan(single_image)
    output, _, _, _ = full_model(single_image) # output, masks, quants, emb_losses

print(output.size())

print("------------- FULL MODEL FLOPS -------------")
macs, params = profile(full_model, inputs=(single_image, ))
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
