import timeit

general_setup = lambda img_size: f"""
import torch
import kornia
from torchvision.transforms import transforms

batch_size = 1
in_tensor = torch.randn(batch_size, 3, {img_size}, {img_size}).to('cuda:0')
in_pil = transforms.ToPILImage()(in_tensor.cpu().squeeze())

"""

perspective_setup = """
tv_fn = transforms.RandomPerspective(p=1.0)
kor_fn = kornia.augmentation.RandomPerspective(p=1.0)
"""

colorjitter_setup = """
tv_fn = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
kor_fn = kornia.augmentation.ColorJitter(0.5, 0.5, 0.5, 0.5)
"""

affine_setup = """
tv_fn = transforms.RandomAffine(360, (0.1, 0.1), (1.1, 1.2), (0.1, 0.1), resample=2)
kor_fn = kornia.augmentation.RandomAffine(360, (0.1, 0.1), (1.1, 1.2), (0.1, 0.1))
"""

vflip_setup = """
tv_fn = transforms.RandomVerticalFlip(p=1.0)
kor_fn = kornia.augmentation.RandomVerticalFlip(p=1.0)
"""

hflip_setup = """
tv_fn = transforms.RandomHorizontalFlip(p=1.0)
kor_fn = kornia.augmentation.RandomHorizontalFlip(p=1.0)
"""

rotate_setup = """
tv_fn = transforms.RandomRotation(180, resample=2)
kor_fn = kornia.augmentation.RandomRotation(180.)
"""

crop_setup = """
tv_fn = transforms.RandomCrop((180, 180))
kor_fn = kornia.augmentation.RandomCrop((180, 180))
"""

erasing_setup = """
tv_fn = transforms.RandomErasing(p=1.0)
kor_fn = kornia.augmentation.RandomErasing((0.02, 0.33), (0.3, 3.3))
"""

grayscale_setup = """
tv_fn = transforms.RandomGrayscale(p=1.0)
kor_fn = kornia.augmentation.RandomGrayscale(p=1.0)
"""

res_crop_setup = """
tv_fn = transforms.RandomResizedCrop((180, 180))
kor_fn = kornia.augmentation.RandomResizedCrop((180, 180), (0.08, 1.0), (0.75, 1.33))
"""

cent_crop_setup = """
tv_fn = transforms.CenterCrop((180, 180))
kor_fn = kornia.augmentation.CenterCrop((180, 180))
"""


if __name__ == '__main__':
    image_sizes = {
        # Coefficients:   size
        'efficientnet-b0': 224,
        'efficientnet-b1': 240,
        'efficientnet-b2': 260,
        'efficientnet-b3': 300,
        'efficientnet-b4': 380,
        'efficientnet-b5': 456,
        'efficientnet-b6': 528,
        'efficientnet-b7': 600,
        'efficientnet-b8': 672,
        'efficientnet-l2': 800,
    }
    fn_set_ups = {
        'perspective_setup': perspective_setup,
        'colorjitter_setup': colorjitter_setup,
        'affine_setup': affine_setup,
        'vflip_setup': vflip_setup,
        'hflip_setup': hflip_setup,
        'rotate_setup': rotate_setup,
        'crop_setup': crop_setup,
        'erasing_setup': erasing_setup,
        'grayscale_setup': grayscale_setup,
        'res_crop_setup': res_crop_setup,
        'cent_crop_setup': cent_crop_setup
    }

    num = 100
    for name in fn_set_ups:
        for net in image_sizes:
            tv = timeit.timeit(
                "tv_fn(in_tensor.squeeze())" if name == 'erasing_setup' else "tv_fn(in_pil)" ,
                setup=general_setup(image_sizes[net]) + fn_set_ups[name],
                number=num
            )

            kor = timeit.timeit(
                "kor_fn(in_tensor)",
                setup=general_setup(image_sizes[net]) + fn_set_ups[name],
                number=num
            )
            print(name, net, "image_size=%d" % image_sizes[net], "Torchvision: %.4fs" % (tv/num), \
                "Kornia: %.4fs" % (kor/num), "Win" if kor < tv else "Lose")
