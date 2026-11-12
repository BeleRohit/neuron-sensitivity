from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import TensorDataset


FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def _download(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    url = BASE_URL + path.name
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download MNIST file {url}: {exc}") from exc


def _read_images(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as stream:
        magic, count, rows, cols = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file: {path}")
        data = torch.frombuffer(bytearray(stream.read()), dtype=torch.uint8)
    return data.reshape(count, 1, rows, cols).float().div_(255)


def _read_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file: {path}")
        data = torch.frombuffer(bytearray(stream.read()), dtype=torch.uint8)
    return data.reshape(count).long()


def load_mnist(root: str | Path = "data") -> tuple[TensorDataset, TensorDataset]:
    root = Path(root)
    paths = {key: root / filename for key, filename in FILES.items()}
    for path in paths.values():
        if not path.exists():
            _download(path)
    return (
        TensorDataset(_read_images(paths["train_images"]), _read_labels(paths["train_labels"])),
        TensorDataset(_read_images(paths["test_images"]), _read_labels(paths["test_labels"])),
    )
