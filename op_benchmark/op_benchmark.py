import timeit
import os
import numpy as np
import pandas as pd


general_setup = lambda batch_size, img_size: f"""
import torch
import kornia
from torchvision.transforms import transforms

in_tensor = torch.randn({batch_size}, 3, {img_size}, {img_size}).to('cuda:0')
in_pil = transforms.ToPILImage()(in_tensor.cpu()[0])

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
kor_fn = kornia.augmentation.RandomErasing(1.0, (0.02, 0.33), (0.3, 3.3))
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
    save_to = "data"
    num = 10

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
    }
    fn_set_ups = {
        'RandomPerspective': perspective_setup,
        'ColorJitter': colorjitter_setup,
        'RandomAffine': affine_setup,
        'RandomVerticalFlip': vflip_setup,
        'RandomHorizontalFlip': hflip_setup,
        'RandomRotate': rotate_setup,
        'RandomCrop': crop_setup,
        'RandomErasing': erasing_setup,
        'RandomGrayscale': grayscale_setup,
        'RandomResizedCrop': res_crop_setup,
        'RandomCenterCrop': cent_crop_setup
    }

    tv_timer = lambda batch_size, net, name: timeit.Timer(
        f"[tv_fn({'in_tensor[0].squeeze()' if name == 'RandomErasing' else 'in_pil'}) for _ in range({batch_size})]" ,
        setup=general_setup(batch_size, image_sizes[net]) + fn_set_ups[name]
    )
    
    kor_timer = lambda batch_size, net, name: timeit.Timer(
        "kor_fn(in_tensor)",
        setup=general_setup(batch_size, image_sizes[net]) + fn_set_ups[name]
    )

    try:
        os.mkdir(save_to)
    except:
        pass

    for timer_name, batch_size, timer in [
          ("torchvision", 1, tv_timer),
          ("kornia", 1, kor_timer),
          ("kornia", 2, kor_timer),
          ("kornia", 4, kor_timer),
          ("kornia", 8, kor_timer),
          ("kornia", 16, kor_timer),
          ("kornia", 32, kor_timer),
          ("kornia", 64, kor_timer)
    ]:
        dfs = []
        for name in fn_set_ups:
            row_res = {'op_name': name}
            for net in image_sizes:
                timer = kor_timer(batch_size, net, name)
                timer_res = timer.repeat(num, 1)

                row_res.update({net: np.mean(timer_res) * 1000})
                print(name, net, "image_size=%d" % image_sizes[net], "batchsize=%d" % batch_size, "%.2f±%.2fms" % (np.mean(timer_res) * 1000, np.std(timer_res) * 1000))
            dfs.append(pd.DataFrame.from_dict(row_res, orient='index').T)
        df = pd.concat(dfs)
        df.to_csv(f"{save_to}/{timer_name}_bs{batch_size}.csv", index=None)
