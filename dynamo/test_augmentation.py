import pickle
from datetime import datetime
from itertools import product

import torch
import torch.utils.benchmark as benchmark
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
from plot import graphs_from_results
from runner import _check_run
from runner import _iter_op_device
from runner import _unpick


torch.set_float32_matmul_precision('high')

# ----------------------------------------------------------------------
# Config benchmark
# ----------------------------------------------------------------------
batch_sizes = [1, 8, 32]  # 64, 128, 256]
resolutions = [128, 254]
dtypes = [torch.float32, torch.float64]
devices = ['cpu', 'cuda']
threads = [1, 8]

# List Augmentations
tfms_to_bench = {
    # not crashing on dynamo on kornia@3f2361eaf1691288789f17b7b7daa3d5a0866003
    'RandomAffine': 'randomaffine',
    # 'ColorJiggle': 'colorjiggle',
    # 'RandomBoxBlur': 'randomboxblur',
    # 'RandomBrightness': 'randombrightness',
    # 'RandomChannelShuffle': 'randomchannelshuffle',
    # 'RandomContrast': 'randomcontrast',
    # 'RandomEqualize': 'randomequalize',
    # 'RandomGamma': 'randomgamma',
    # 'RandomGaussianBlur': 'randomgaussianblur',
    # 'RandomGaussianNoise': 'randomgaussiannoise',
    # 'RandomHue': 'randomhue',
    # 'RandomRGBShift': 'randomrgbshift',
    # 'RandomSaturation': 'randomsaturation',
    # 'RandomSharpness': 'randomsharpness',
    # 'RandomSolarize': 'randomsolarize',
    # 'PadTo': 'padto',
    # 'RandomCrop': 'randomcrop',
    # 'RandomErasing': 'randomerasing',
    # 'RandomFisheye': 'randomfisheye',
    # 'RandomInvert': 'randominvert',
    # 'RandomPerspective': 'randomperspective',
    # 'RandomResizedCrop': 'randomresizedcrop',
    # 'RandomRotation': 'randomrotation',
    # 'RandomVerticalFlip': 'randomverticalflip',


    # not crashing but really worst performance
    # 'RandomPlanckianJitter': 'randomplanckianjitter',
    # "CenterCrop": "centercrop",
    # "RandomHorizontalFlip": "randomhorizontalflip",
    # 'RandomJigsaw': 'randomjigsaw',


    # crashing ;/
    # "RandomMotionBlur": "randommotionblur",
    # "RandomMixUpV2": "randommixupv2",
    # "RandomMosaic": "randommosaic",
    # "RandomCutMixV2": "randomcutmixv2",

    # crashing on cpu
    # "ColorJitter": "colorjitter",
    # "RandomGrayscale": "randomgrayscale",


    # crashing on cuda
    # "RandomPosterize": "randomposterize",

    # Needs to check
    # "RandomPlasmaShadow": "randomplasmashadow",
    # "RandomPlasmaBrightness": "randomplasmabrightness",
    # "RandomPlasmaContrast": "randomplasmacontrast",
    # "RandomElasticTransform": "randomelastictransform",
    # "RandomThinPlateSpline": "randomthinplatespline",
}

# ----------------------------------------------------------------------
# Init augmentation modules
# ----------------------------------------------------------------------
# Intensity 2D transforms
randomplanckianjitter = RandomPlanckianJitter(
    'blackbody', same_on_batch=False, keepdim=False, p=1.0,
)
randomplasmashadow = RandomPlasmaShadow(
    roughness=(0.1, 0.7), shade_intensity=(-1.0, 0.0),
    shade_quantity=(0.0, 1.0), same_on_batch=False, keepdim=False, p=1.0,
)
randomplasmabrightness = RandomPlasmaBrightness(
    roughness=(0.1, 0.7), intensity=(0.0, 1.0), same_on_batch=False,
    keepdim=False, p=1.0,
)
randomplasmacontrast = RandomPlasmaContrast(
    roughness=(
        0.1, 0.7,
    ), same_on_batch=False, keepdim=False, p=1.0,
)
colorjiggle = ColorJiggle(
    0.3, 0.3, 0.3, 0.3, same_on_batch=False, keepdim=False, p=1.0,
)
colorjitter = ColorJitter(
    0.3, 0.3, 0.3, 0.3, same_on_batch=False, keepdim=False, p=1.0,
)
randomboxblur = RandomBoxBlur(
    (21, 21), 'reflect', same_on_batch=False, keepdim=False, p=1.0,
)
randombrightness = RandomBrightness(
    brightness=(
        0.8, 1.2,
    ), clip_output=True, same_on_batch=False, keepdim=False, p=1.0,
)
randomchannelshuffle = RandomChannelShuffle(
    same_on_batch=False, keepdim=False, p=1.0,
)
randomcontrast = RandomContrast(
    contrast=(0.8, 1.2), clip_output=True, same_on_batch=False,
    keepdim=False, p=1.0,
)
randomequalize = RandomEqualize(same_on_batch=False, keepdim=False, p=1.0)
randomgamma = RandomGamma(
    (0.2, 1.3), (1.0, 1.5),
    same_on_batch=False, keepdim=False, p=1.0,
)
randomgrayscale = RandomGrayscale(
    same_on_batch=False, keepdim=False, p=1.0,
)
randomgaussianblur = RandomGaussianBlur(
    (21, 21), (0.2, 1.3), 'reflect', same_on_batch=False, keepdim=False,
    p=1.0, silence_instantiation_warning=True,
)
randomgaussiannoise = RandomGaussianNoise(
    mean=0.2, std=0.7, same_on_batch=False, keepdim=False, p=1.0,
)
randomhue = RandomHue((-0.2, 0.4), same_on_batch=False, keepdim=False, p=1.0)
randommotionblur = RandomMotionBlur(
    (7, 7), 5.0, 0.5, 'reflect', 'nearest', same_on_batch=False, keepdim=False,
    p=1.0,
)
randomposterize = RandomPosterize(
    bits=3, same_on_batch=False, keepdim=False, p=1.0,
)
randomrgbshift = RandomRGBShift(
    r_shift_limit=0.5, g_shift_limit=0.5, b_shift_limit=0.5,
    same_on_batch=False, keepdim=False, p=1.0,
)
randomsaturation = RandomSaturation(
    (0.5, 1.0), same_on_batch=False, keepdim=False, p=1.0,
)
randomsharpness = RandomSharpness(
    (0.5, 1.0), same_on_batch=False, keepdim=False, p=1.0,
)
randomsolarize = RandomSolarize(
    0.1, 0.1, same_on_batch=False, keepdim=False, p=1.0,
)

# Geometric 2d transforms
centercrop = CenterCrop(
    64, resample='nearest', cropping_mode='resample',
    align_corners=True, keepdim=False, p=1.0,
)
padto = PadTo((500, 500), 'constant', 1, keepdim=False)
randomaffine = RandomAffine(
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
)
randomcrop = RandomCrop(
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
)
randomerasing = RandomErasing(
    scale=(0.02, 0.33), ratio=(
        0.3, 3.3,
    ), value=1, same_on_batch=False, keepdim=False, p=1.0,
)
randomelastictransform = RandomElasticTransform(
    (27, 27), (33, 31), (0.5, 1.5), align_corners=True,
    padding_mode='reflection', same_on_batch=False, keepdim=False, p=1.0,
)
c = torch.tensor([-0.3, 0.3])
g = torch.tensor([0.9, 1.0])
randomfisheye = RandomFisheye(
    c, c, g, same_on_batch=False, keepdim=False, p=1.0,
)
randomhorizontalflip = RandomHorizontalFlip(
    same_on_batch=False, keepdim=False, p=1.0,
)
randominvert = RandomInvert(same_on_batch=False, keepdim=False, p=1.0)
randomperspective = RandomPerspective(
    0.5, 'nearest', align_corners=True, same_on_batch=False,
    keepdim=False, p=1.0,
)
randomresizedcrop = RandomResizedCrop(
    (200, 200),
    (0.4, 1.0),
    (2.0, 2.0),
    'nearest',
    align_corners=True,
    cropping_mode='resample',
    same_on_batch=False,
    keepdim=False,
    p=1.0,
)
randomrotation = RandomRotation(
    15.0, 'nearest', align_corners=True, same_on_batch=False,
    keepdim=False, p=1.0,
)
randomverticalflip = RandomVerticalFlip(
    same_on_batch=False, keepdim=False, p=1.0, p_batch=0.5,
)
randomthinplatespline = RandomThinPlateSpline(
    0.3, align_corners=True, same_on_batch=False, keepdim=False, p=1.0,
)

# Mix 2d transforms
randomcutmixv2 = RandomCutMixV2(
    1, (0.2, 0.9), same_on_batch=False, keepdim=False, p=1.0,
)
randommixupv2 = RandomMixUpV2(
    (0.1, 0.9), same_on_batch=False, keepdim=False, p=1.0,
)
randommosaic = RandomMosaic(
    (250, 125),
    (4, 4),
    (0.3, 0.7),
    align_corners=True,
    cropping_mode='resample',
    padding_mode='reflect',
    resample='nearest',
    keepdim=False,
    p=1.0,
)
randomjigsaw = RandomJigsaw(
    (2, 2), ensure_perm=True, same_on_batch=False, keepdim=False, p=1.0,
)


# ----------------------------------------------------------------------
# bench
# ----------------------------------------------------------------------
results = []
_dt = datetime.strftime(datetime.utcnow(), '%Y%m%d_%H%M%S')
filename = f'output-benchmark-augmentation-{_dt}.pickle'
fp = open(filename, 'wb')


for default_op, _, device, optimize in _iter_op_device():
    if 'opencv' in default_op:
        continue

    optz_name, optimizer_txt, optimizer = optimize
    dev = torch.device(device)
    device_name = torch.cuda.get_device_name(0)

    print('\n\n', '-'*79, f'\nBenchmarking {optimizer_txt} on {device}...')

    cases = product(resolutions, dtypes, batch_sizes, threads)
    for res, dtype, bs, num_threads in cases:

        dtype_str = str(dtype).split('.')[-1]
        print(f'\n\t{res=}-{bs=}-dtype={dtype_str}')

        x = torch.rand(bs, 3, res, res, device=dev, dtype=dtype)

        sub_label = f'[{bs}, {res}, {dtype_str}]'

        for aug_name, op_name in tfms_to_bench.items():
            print(f'\t\t {aug_name}...')
            if not _check_run(True, '__main__', op_name, x, optimizer):
                print('\t\t\tSkipping bench because it did not work!')
                continue

            global op_to_bench
            op_to_bench = eval(op_name)
            if optimizer:
                op_to_bench = optimizer(op_to_bench)

            bench_out = benchmark.Timer(
                stmt='op_to_bench(input)',
                setup='from __main__ import op_to_bench',
                globals={'input': x},
                num_threads=num_threads,
                label=aug_name,
                sub_label=sub_label,
                description=f'{optz_name}kornia_{device}',
            ).blocked_autorange(min_run_time=1)

            pickle.dump(bench_out, fp, protocol=pickle.HIGHEST_PROTOCOL)


# ----------------------------------------------------------------------
# Compare results
# ----------------------------------------------------------------------
results = _unpick(filename)
compare = benchmark.Compare(results)
compare.print()


# ----------------------------------------------------------------------
# Generate the graphs
# ----------------------------------------------------------------------
graphs_from_results(results)
