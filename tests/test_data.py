import struct

from neuron_sensitivity.data import load_mnist


def test_load_mnist_reads_raw_idx_files_without_network(tmp_path):
    raw = tmp_path / "MNIST" / "raw"
    raw.mkdir(parents=True)
    (raw / "train-images-idx3-ubyte").write_bytes(
        struct.pack(">IIII", 2051, 1, 28, 28) + bytes([128]) * (28 * 28)
    )
    (raw / "t10k-images-idx3-ubyte").write_bytes(
        struct.pack(">IIII", 2051, 1, 28, 28) + bytes([255]) * (28 * 28)
    )
    (raw / "train-labels-idx1-ubyte").write_bytes(struct.pack(">II", 2049, 1) + b"\x03")
    (raw / "t10k-labels-idx1-ubyte").write_bytes(struct.pack(">II", 2049, 1) + b"\x07")

    train, test = load_mnist(tmp_path)
    train_image, train_label = train[0]
    test_image, test_label = test[0]
    assert tuple(train_image.shape) == (1, 28, 28)
    assert abs(train_image[0, 0, 0].item() - 128 / 255) < 1e-6
    assert train_label.item() == 3
    assert test_image[0, 0, 0].item() == 1.0
    assert test_label.item() == 7
