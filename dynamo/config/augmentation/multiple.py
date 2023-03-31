import torch
from kornia.augmentation import CenterCrop
from kornia.augmentation import ColorJiggle
from kornia.augmentation import ColorJitter
from kornia.augmentation import PadTo
from kornia.augmentation import RandomAffine
from kornia.augmentation import RandomBoxBlur
from kornia.augmentation import RandomBrightness
from kornia.augmentation import RandomChannelShuffle
from kornia.augmentation import RandomContrast
from kornia.augmentation import RandomCrop
from kornia.augmentation import RandomCutMixV2
from kornia.augmentation import RandomElasticTransform
from kornia.augmentation import RandomEqualize
from kornia.augmentation import RandomErasing
from kornia.augmentation import RandomFisheye
from kornia.augmentation import RandomGamma
from kornia.augmentation import RandomGaussianBlur
from kornia.augmentation import RandomGaussianNoise
from kornia.augmentation import RandomGrayscale
from kornia.augmentation import RandomHorizontalFlip
from kornia.augmentation import RandomHue
from kornia.augmentation import RandomInvert
from kornia.augmentation import RandomJigsaw
from kornia.augmentation import RandomMixUpV2
from kornia.augmentation import RandomMosaic
from kornia.augmentation import RandomMotionBlur
from kornia.augmentation import RandomPerspective
from kornia.augmentation import RandomPlanckianJitter
from kornia.augmentation import RandomPlasmaBrightness
from kornia.augmentation import RandomPlasmaContrast
from kornia.augmentation import RandomPlasmaShadow
from kornia.augmentation import RandomPosterize
from kornia.augmentation import RandomResizedCrop
from kornia.augmentation import RandomRGBShift
from kornia.augmentation import RandomRotation
from kornia.augmentation import RandomSaturation
from kornia.augmentation import RandomSharpness
from kornia.augmentation import RandomSolarize
from kornia.augmentation import RandomThinPlateSpline
from kornia.augmentation import RandomVerticalFlip
from kornia.core import Tensor


c = torch.tensor([-0.3, 0.3])
g = torch.tensor([0.9, 1.0])

# List Augmentations
KORNIA_OPS = {
    # not crashing on dynamo on kornia@3f2361eaf1691288789f17b7b7daa3d5a0866003
    'RandomAffine': RandomAffine(
        (-15.0, 5.0),
        (0.3, 1.0),
        (0.4, 1.3),
        0.5,
        resample='nearest',
        padding_mode='reflection',
        align_corners=True,
        same_on_batch=False,
        keepdim=False,
        p=1.0,
    ),
    'ColorJiggle': ColorJiggle(
        0.3, 0.3, 0.3, 0.3, same_on_batch=False, keepdim=False, p=1.0,
    ),
    'RandomBoxBlur':  RandomBoxBlur(
        (21, 21), 'reflect', same_on_batch=False, keepdim=False, p=1.0,
    ),
    'RandomBrightness': RandomBrightness(
        brightness=(0.8, 1.2), clip_output=True, same_on_batch=False,
        keepdim=False, p=1.0,
    ),
    'RandomChannelShuffle': RandomChannelShuffle(
        same_on_batch=False, keepdim=False, p=1.0,
    ),
    'RandomContrast': RandomContrast(
        contrast=(0.8, 1.2), clip_output=True, same_on_batch=False,
        keepdim=False, p=1.0,
    ),
    'RandomEqualize': RandomEqualize(
        same_on_batch=False, keepdim=False, p=1.0,
    ),
    'RandomGamma': RandomGamma(
        (0.2, 1.3), (1.0, 1.5),
        same_on_batch=False, keepdim=False, p=1.0,
    ),
    'RandomGaussianBlur': RandomGaussianBlur(
        (21, 21), (0.2, 1.3), 'reflect', same_on_batch=False, keepdim=False,
        p=1.0, silence_instantiation_warning=True,
    ),
    'RandomGaussianNoise': RandomGaussianNoise(
        mean=0.2, std=0.7, same_on_batch=False, keepdim=False, p=1.0,
    ),
    'RandomHue': RandomHue(
        (-0.2, 0.4), same_on_batch=False, keepdim=False, p=1.0,
    ),
    'RandomRGBShift': RandomRGBShift(
        r_shift_limit=0.5, g_shift_limit=0.5, b_shift_limit=0.5,
        same_on_batch=False, keepdim=False, p=1.0,
    ),
    'RandomSaturation': RandomSaturation(
        (0.5, 1.0), same_on_batch=False, keepdim=False, p=1.0,
    ),
    'RandomSharpness': RandomSharpness(
        (0.5, 1.0), same_on_batch=False, keepdim=False, p=1.0,
    ),
    'RandomSolarize': RandomSolarize(
        0.1, 0.1, same_on_batch=False, keepdim=False, p=1.0,
    ),
    'PadTo': PadTo((500, 500), 'constant', 1, keepdim=False),
    'RandomCrop': RandomCrop(
        (150, 150),
        10,
        True,
        1,
        'constant',
        'nearest',
        cropping_mode='resample',
        same_on_batch=False,
        align_corners=True,
        keepdim=False,
        p=1.0,
    ),
    'RandomErasing': RandomErasing(
        scale=(0.02, 0.33), ratio=(0.3, 3.3), value=1, same_on_batch=False,
        keepdim=False, p=1.0,
    ),
    'RandomFisheye': RandomFisheye(
        c, c, g, same_on_batch=False, keepdim=False, p=1.0,
    ),
    'RandomInvert': RandomInvert(same_on_batch=False, keepdim=False, p=1.0),
    'RandomPerspective': RandomPerspective(
        0.5, 'nearest', align_corners=True, same_on_batch=False,
        keepdim=False, p=1.0,
    ),
    'RandomResizedCrop': RandomResizedCrop(
        (200, 200),
        (0.4, 1.0),
        (2.0, 2.0),
        'nearest',
        align_corners=True,
        cropping_mode='resample',
        same_on_batch=False,
        keepdim=False,
        p=1.0,
    ),
    'RandomRotation': RandomRotation(
        15.0, 'nearest', align_corners=True, same_on_batch=False,
        keepdim=False, p=1.0,
    ),
    'RandomVerticalFlip': RandomVerticalFlip(
        same_on_batch=False, keepdim=False, p=1.0, p_batch=0.5,
    ),


    # not crashing but really worst performance
    'RandomPlanckianJitter': RandomPlanckianJitter(
        'blackbody', same_on_batch=False, keepdim=False, p=1.0,
    ),
    'CenterCrop': CenterCrop(
        64, resample='nearest', cropping_mode='resample',
        align_corners=True, keepdim=False, p=1.0,
    ),
    'RandomHorizontalFlip': RandomHorizontalFlip(
        same_on_batch=False, keepdim=False, p=1.0,
    ),
    'RandomJigsaw': RandomJigsaw(
        (2, 2), ensure_perm=True, same_on_batch=False, keepdim=False, p=1.0,
    ),


    # crashing ;/
    'RandomMotionBlur': RandomMotionBlur(
        (7, 7), 5.0, 0.5, 'reflect', 'nearest', same_on_batch=False,
        keepdim=False, p=1.0,
    ),
    'RandomMixUpV2': RandomMixUpV2(
        (0.1, 0.9), same_on_batch=False, keepdim=False, p=1.0,
    ),
    'RandomMosaic': RandomMosaic(
        (250, 125),
        (4, 4),
        (0.3, 0.7),
        align_corners=True,
        cropping_mode='resample',
        padding_mode='reflect',
        resample='nearest',
        keepdim=False,
        p=1.0,
    ),
    'RandomCutMixV2': RandomCutMixV2(
        1, (0.2, 0.9), same_on_batch=False, keepdim=False, p=1.0,
    ),

    # crashing on cpu
    'ColorJitter': ColorJitter(
        0.3, 0.3, 0.3, 0.3, same_on_batch=False, keepdim=False, p=1.0,
    ),
    'RandomGrayscale': RandomGrayscale(
        same_on_batch=False, keepdim=False, p=1.0,
    ),


    # crashing on cuda
    'RandomPosterize': RandomPosterize(
        bits=3, same_on_batch=False, keepdim=False, p=1.0,
    ),

    # Needs to check
    'RandomPlasmaShadow': RandomPlasmaShadow(
        roughness=(0.1, 0.7), shade_intensity=(-1.0, 0.0),
        shade_quantity=(0.0, 1.0), same_on_batch=False, keepdim=False, p=1.0,
    ),
    'RandomPlasmaBrightness':  RandomPlasmaBrightness(
        roughness=(0.1, 0.7), intensity=(0.0, 1.0), same_on_batch=False,
        keepdim=False, p=1.0,
    ),
    'RandomPlasmaContrast': RandomPlasmaContrast(
        roughness=(0.1, 0.7), same_on_batch=False, keepdim=False, p=1.0,
    ),
    'RandomElasticTransform': RandomElasticTransform(
        (27, 27), (33, 31), (0.5, 1.5), align_corners=True,
        padding_mode='reflection', same_on_batch=False, keepdim=False, p=1.0,
    ),
    'RandomThinPlateSpline': RandomThinPlateSpline(
        0.3, align_corners=True, same_on_batch=False, keepdim=False, p=1.0,
    ),
}


def kornia_op(input: Tensor, *args, **kwargs):
    mode = kwargs['mode']
    del kwargs['mode']

    KORNIA_OPS[mode](input, *args, **kwargs)


def opencv_op(*args, **kwargs) -> None:
    raise NotImplementedError('We do not have augmentations on opencv')
