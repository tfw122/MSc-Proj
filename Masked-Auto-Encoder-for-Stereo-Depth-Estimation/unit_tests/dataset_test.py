#from src.utils.builder import build_trainer, build_config
from src.utils.utils import *
from src.utils.builder import *
from src.utils.fileio import *
from src.common.registry import registry
from arguments import args
from time import time
from PIL import Image
import numpy as np
from torchvision.utils import save_image, make_grid

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    
def get_transforms(config, split):
    transforms_name = config.dataset_config.preprocess.name
    data_transform_cls = registry.get_preprocessor_class(transforms_name)
    data_transforms_obj = data_transform_cls(config, split)
    return data_transforms_obj

def ToArray(img_t):
    img = img_t.detach().to("cpu").numpy()
    img = np.transpose(img, (1, 2, 0))

    return img

# --------------- configure initialisers -----------------
setup_imports()
parser= args.get_parser()
opts, unknown = parser.parse_known_args()

config = build_config(opts)

# setup aws env variables:
fileio_client = FileIOClient(config)

# check correct config loaded;
print(config.dataset_config.dataset_name)
#print(config.dataset_config.usable_columns)

# --------------- Import Dataset Builder function -----------------
from src.datasets.dataset_zoo.stereo_vision_mim.stereo_datasets_mim.cityscapes_dataset import *
from src.datasets.dataset_zoo.stereo_vision_mim.stereo_datasets_mim.kitti_dataset import *
from src.datasets.dataset_zoo.stereo_vision_mim.stereo_datasets_mim.falling_things_dataset import *
from src.datasets.dataset_zoo.stereo_vision_mim.stereo_datasets_mim.sceneflow_dataset import *
from src.datasets.dataset_zoo.stereo_vision_mim.stereo_datasets_mim.sintel_stereo_dataset import *
from src.datasets.dataset_zoo.stereo_vision_mim.stereo_datasets_mim.tartan_air_dataset import *

from src.datasets.dataset_zoo.stereo_vision_mim.stereo_vision_mim_builder import StereoVisionMaskedImageModellingDatasetModule

from torchvision.transforms import transforms

train_transforms = get_transforms(config, 'train')
test_transforms = get_transforms(config, 'test')

# -------------- Initialise dataset -----------------
train_dataset = CityScapesLoader(config, 'train', train_transforms)
val_dataset = CityScapesLoader(config, 'val', train_transforms)

# iterate over the dataset
print('no. training # samples: {}'.format(len(train_dataset)))
print('no. val # samples: {}'.format(len(val_dataset)))
# -------------- Iterate over a few samples ------------
for i in range(len(train_dataset)):
    if i > 3:
        break
    else:
        # left_image, right_image = dataset[i]
        sample = train_dataset[i]
        left_image, right_image = sample['left_image'], sample['right_image']
        
        print(i, type(sample))
        print(left_image.size(), right_image.size())
        
        left_img= ToArray(left_img)
        right_img= ToArray(right_img)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('CityScapes Stereo Images')
        ax1.imshow(left_img)
        ax2.imshow(right_img)
        plt.show()
# docker container exec -u 0 -it {your_container_id} bash