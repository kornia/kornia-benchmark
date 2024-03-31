from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.utils.benchmark as benchmark
import torchvision as tv
import kornia as K
import kornia_rs as KR
from accelerate import Accelerator
from tqdm import tqdm
import cv2

DATA_PATH = Path(__file__).parent / "dog.jpeg"


def kornia_preprocess(img_path: Path) -> torch.Tensor:
    img = KR.read_image_jpeg(str(img_path))  # HxWxC
    img = KR.resize(img, (224, 224), interpolation="bilinear")
    img = K.utils.image_to_tensor(img, keepdim=True)  # CxHxW
    img = img.float() / 255.0
    return img


def torch_preprocess(img_path: Path) -> torch.Tensor:
    img = tv.io.read_image(str(img_path))  # HxWxC
    with torch.inference_mode():
        img = tv.transforms.functional.resize(img, (224, 224))
        img = img.float() / 255.0
    return img


def opencv_preprocess(img_path: Path) -> torch.Tensor:
    img = cv2.imread(str(img_path))  # HxWxC
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = K.utils.image_to_tensor(img, keepdim=True)  # CxHxW
    img = img.float() / 255.0
    return img


class MockImageDataset(torch.utils.data.Dataset):
    NUM_SAMPLES = 10000

    def __init__(self, preprocess_fn):
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return self.NUM_SAMPLES

    def __getitem__(self, _) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.preprocess_fn(DATA_PATH)
        label = torch.tensor(0)
        return img, label


def train_one_epoch(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    model.train()
    for data, target in tqdm(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size", type=int, default=32, help="The batch size to use for training."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="The number of workers to use for the dataloader.",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        choices=["kornia", "opencv", "torch"],
        default="kornia_rs",
        help="Preprocessing backend",
    )
    args = parser.parse_args()

    accelerate = Accelerator()

    if args.preprocess == "kornia":
        preprocess_fn = kornia_preprocess
    elif args.preprocess == "opencv":
        preprocess_fn = opencv_preprocess
    else:
        preprocess_fn = torch_preprocess

    dataset = MockImageDataset(preprocess_fn=preprocess_fn)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = tv.models.resnet50(weights=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataloader, model, optimizer = accelerate.prepare(dataloader, model, optimizer)

    num_threads = torch.get_num_threads()
    print(f"Benchmarking in {num_threads} threads")

    timer = benchmark.Timer(
        stmt="train_one_epoch(dataloader, model, optimizer)",
        setup="from __main__ import train_one_epoch",
        globals={"dataloader": dataloader, "model": model, "optimizer": optimizer},
        num_threads=num_threads,
        label=f"Training ResNet50 with {args.preprocess} preprocessing",
        sub_label=f"batch_size={args.batch_size}",
    )

    print(timer.timeit(1))


if __name__ == "__main__":
    main()
