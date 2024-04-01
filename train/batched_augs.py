import argparse
import time
import torch
import kornia as K
import kornia_rs as KR
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batched", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    num_tests = 1000

    times = []

    img = KR.read_image_jpeg("dog.jpeg")

    img_list = np.array([img for _ in range(args.batch_size)])
    img_batch = (
        torch.from_numpy(img_list).to(device="cuda").permute(0, 3, 1, 2).float() / 255.0
    )

    for _ in range(num_tests):
        torch.cuda.synchronize()
        t0 = time.monotonic()
        if args.batched:
            K.geometry.resize(img_batch, (128, 128), "bilinear")
        else:
            for img in img_list:
                KR.resize(img, (128, 128), "bilinear")
        torch.cuda.synchronize()
        t1 = time.monotonic()
        times.append(t1 - t0)

    print(f"Median time: {np.median(times)}s")


if __name__ == "__main__":
    main()
