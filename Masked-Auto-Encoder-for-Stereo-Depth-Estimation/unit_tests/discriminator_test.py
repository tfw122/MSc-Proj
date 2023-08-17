#from src.utils.builder import build_trainer, build_config
from src.utils.utils import *
from src.utils.builder import *
from utils.fileio import *
from arguments import args
from time import time
from PIL import Image
import numpy as np
import torchvision.transforms as transforms 
from src.losses.image_reconstruction import scale_pyramid

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

# --------------- import the discriminator ----------------
from src.models.modules.discriminators import Discriminator, MSGDiscriminator

model = MSGDiscriminator(config) 
# put it in eval mode; batch norm stats will mess up forward pass otherwise
model.eval();
print(model)
print(model.layers[0])
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
    num_scales=6

    rand_multi_image = torch.randn((8,3,n_images,224,224))
    single_image = torch.randn((8,3,224,224))
    pyramid_input = scale_pyramid(single_image, num_scales)

    for i in pyramid_input:
        print(i.size())
    
    print('reversed list;')
    for i in pyramid_input[::-1]:
        print(i.size())
    # ----------------- model inference ----------------
    print('make sure for MSG GAN the input pyramid is reversed; i.e. ascending to descending')
    output = model(pyramid_input[::-1])

# check if output is as expected
print(type(output), output.size())
