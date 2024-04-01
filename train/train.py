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
import albumentations as A

DATA_PATH = Path(__file__).parent / "dog.jpeg"


def kornia_preprocess(img_path: Path) -> torch.Tensor:
    img = KR.read_image_jpeg(str(img_path))  # HxWxC
    img = KR.resize(img, (224, 224), interpolation="bilinear")
    img = K.utils.image_to_tensor(img, keepdim=True)  # CxHxW
    img = img.float() / 255.0
    return img


def opencv_preprocess(img_path: Path) -> torch.Tensor:
    img = cv2.imread(str(img_path))  # HxWxC
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    return img


class MockImageDataset(torch.utils.data.Dataset):
    NUM_SAMPLES = 2000

    def __init__(
        self,
        augmentations_fn: callable | None = None,
        strategy: str = "hybrid",
    ):
        self.augmentations_fn = augmentations_fn
        self.strategy = strategy

    def __len__(self):
        return self.NUM_SAMPLES

    def __getitem__(self, _) -> tuple[torch.Tensor, torch.Tensor]:
        if self.strategy == "hybrid":
            img = kornia_preprocess(DATA_PATH)
        else:
            img = opencv_preprocess(DATA_PATH)
            if self.augmentations_fn is not None:
                out = self.augmentations_fn(image=img)
                img = out["image"]
            img = K.utils.image_to_tensor(img)
            img = img.float() / 255.0

        label = torch.tensor(0)
        return img, label


def train_one_epoch(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    augmentations_fn: torch.nn.Module | None = None,
) -> None:
    model.train()
    for data, target in tqdm(dataloader):
        optimizer.zero_grad()
        if augmentations_fn is not None:
            data = augmentations_fn(data)
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
        "--mixed-precision",
        type=str,
        choices=["no", "fp16"],
        default=None,
        help="Whether to use mixed precision training.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="The number of workers to use for the dataloader.",
    )
    parser.add_argument(
        "--augmentations-strategy",
        type=str,
        choices=["hybrid", "sequential"],
        default="hybrid",
        help="Preprocessing backend",
    )
    args = parser.parse_args()

    accelerate = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    augmentations_fn = None
    if args.augmentations_strategy == "hybrid":
        augmentations_fn = torch.nn.Sequential(
            K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1),
        )
    else:
        augmentations_fn = A.Compose(
            [
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ]
        )

    dataset = MockImageDataset(
        strategy=args.augmentations_strategy,
        augmentations_fn=augmentations_fn,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = tv.models.resnet50(weights=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataloader, model, optimizer, augmentations_fn = accelerate.prepare(
        dataloader,
        model,
        optimizer,
        augmentations_fn,
    )

    num_tests = 25
    num_threads = torch.get_num_threads()

    print(f"Benchmarking in {num_threads} threads")

    timer = benchmark.Timer(
        stmt="train_one_epoch(dataloader, model, optimizer, augmentations_fn)",
        setup="from __main__ import train_one_epoch",
        globals={
            "dataloader": dataloader,
            "model": model,
            "optimizer": optimizer,
            "augmentations_fn": (
                augmentations_fn if args.augmentations_strategy == "hybrid" else None
            ),
        },
        num_threads=num_threads,
        label=f"Training ResNet50 with {args.augmentations_strategy} augmentations",
        sub_label=f"batch_size={args.batch_size}",
    )
    res = timer.timeit(num_tests)

    print(f"Mean:   {res.mean * 1000} ms")
    print(f"Median: {res.median * 1000} ms")


if __name__ == "__main__":
    main()
