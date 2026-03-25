import os
import pickle
import numpy as np
from pathlib import Path
from collections import Counter
import jax.numpy as jnp
import jax.random as jr
import jax.nn

from damped_linoss.data.dataloader import BaseDataloader, StandardDataloader, BucketedDataloader
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# =============================================
# SECTION: Utility functions
# =============================================


def get_subfolders(folder):
    if os.path.exists(folder):
        return [f.name for f in os.scandir(folder) if f.is_dir()]
    return []


def split(data, bounds: list):
    assert all([b < 1 for b in bounds])
    n = len(data)
    bounds = [0] + [int(n * b) for b in bounds] + [n]
    split_data = [data[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]
    return tuple(split_data)


def shuffle(
    data: tuple,
    labels: tuple,
    key: jax.Array,
    val_proportion: float = 0.15,
    test_proportion: float = 0.15,
) -> tuple[tuple, tuple]:
    """
    Shuffles data ordering and re-splits based on proportion kwargs.

    Args:
        data (tuple): (train_data, val_data, test_data)
        labels (tuple): (train_labels, val_labels, test_labels)
        key (jax.Array): Randomization key, from jax.random.key().
        val_proportion (float): The proportion of the dataset for to validation.
        test_proportion (float): The proportion of the dataset for to test.

    Returns:
        (tuple): (train_data, val_data, test_data),
                    (train_labels, val_labels, test_labels)
    """
    train_data, val_data, test_data = data
    train_labels, val_labels, test_labels = labels

    permutation_key, key = jr.split(key)
    idxs = jr.permutation(
        permutation_key, len(train_data) + len(val_data) + len(test_data)
    )
    if isinstance(train_data, jnp.ndarray) or isinstance(train_data, np.ndarray):
        full_data = jnp.concatenate((train_data, val_data, test_data), axis=0)
        shuffled_data = full_data[idxs]
    else:
        full_data = train_data + val_data + test_data
        shuffled_data = [full_data[i] for i in idxs.tolist()]
    if isinstance(train_labels, jnp.ndarray) or isinstance(
        train_labels, np.ndarray
    ):
        full_labels = jnp.concatenate(
            (train_labels, val_labels, test_labels), axis=0
        )
        shuffled_labels = full_labels[idxs]
    else:
        full_labels = train_labels + val_labels + test_labels
        shuffled_labels = [full_labels[i] for i in idxs.tolist()]

    bounds = [1.0 - val_proportion - test_proportion, 1.0 - test_proportion]
    data = split(shuffled_data, bounds)
    labels = split(shuffled_labels, bounds)

    return data, labels


def append_time(data: tuple, time_duration: float) -> tuple:
    """
    Appends a linearly interpolated time vector to start of arrays.

    Args:
        data (tuple): (train_data, val_data, test_data)
        time_duration (float): Time vector interpolated from 0 to this value.

    Returns:
        (tuple): (train_data, val_data, test_data)
    """
    train_data, val_data, test_data = data

    if isinstance(train_data, list):
        raise NotImplementedError(
            "Including time vector for variable length sequences not implemented."
        )
    else:
        num_timesteps = train_data.shape[1]
        time = jnp.linspace(0, time_duration, num=num_timesteps, endpoint=False)
        train_time = jnp.repeat(time[np.newaxis, ...], len(train_data), axis=0)[
            ..., np.newaxis
        ]
        train_data = jnp.concatenate((train_time, train_data), axis=2)
        val_time = jnp.repeat(time[np.newaxis, ...], len(val_data), axis=0)[
            ..., np.newaxis
        ]
        val_data = jnp.concatenate((val_time, val_data), axis=2)
        test_time = jnp.repeat(time[np.newaxis, ...], len(test_data), axis=0)[
            ..., np.newaxis
        ]
        test_data = jnp.concatenate((test_time, test_data), axis=2)

    return (train_data, val_data, test_data)


def calculate_dimension(data: tuple, labels: tuple, classification: bool) -> tuple[int, int]:
    train_data, _, _ = data
    train_labels, _, _ = labels

    data_dim = train_data[0].shape[1] if train_data[0].ndim == 2 else 1

    # 1D sample could mean (n,1) or (1,n)
    if train_labels[0].ndim == 1:
        if classification:
            label_dim = len(train_labels[0])
        else:
            label_dim = 1
    else:
        label_dim = train_labels[0].shape[-1]

    return data_dim, label_dim


# =============================================
# SECTION: Dataset Class Definition
# =============================================


class Dataset:
    def __init__(
        self,
        name: str,
        data: tuple,
        labels: tuple,
        dataloader_type: type[BaseDataloader],
        data_dim: int,
        label_dim: int,
        in_memory: bool,
        data_out_func: callable,
    ):
        self.name = name
        self.data_dim = data_dim
        self.label_dim = label_dim
        self.in_memory = in_memory
        self.data_out_func = data_out_func

        (train_data, val_data, test_data) = data
        (train_labels, val_labels, test_labels) = labels

        train_loader = dataloader_type(
            train_data, train_labels, self.in_memory, self.data_out_func,
        )
        val_loader = dataloader_type(
            val_data, val_labels, self.in_memory, self.data_out_func
        )
        test_loader = dataloader_type(
            test_data, test_labels, self.in_memory, self.data_out_func
        )

        self.dataloaders = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
    

# =============================================
# SECTION: Dataset-specific generators
# =============================================


def load_UEA_dataset(name, data_dir):
    with open(data_dir + f"/processed/UEA/{name}/data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(data_dir + f"/processed/UEA/{name}/labels.pkl", "rb") as f:
        labels = pickle.load(f)
    onehot_labels = jnp.zeros((len(labels), len(jnp.unique(labels))))
    onehot_labels = onehot_labels.at[jnp.arange(len(labels)), labels].set(1)

    bounds = [0.7, 0.85]
    split_data = split(data, bounds)
    split_labels = split(onehot_labels, bounds)

    return split_data, split_labels, lambda x: x


def load_PPG_dataset(data_dir):
    with open(data_dir + "/processed/PPG/ppg/X_train.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(data_dir + "/processed/PPG/ppg/y_train.pkl", "rb") as f:
        train_labels = pickle.load(f)
    with open(data_dir + "/processed/PPG/ppg/X_val.pkl", "rb") as f:
        val_data = pickle.load(f)
    with open(data_dir + "/processed/PPG/ppg/y_val.pkl", "rb") as f:
        val_labels = pickle.load(f)
    with open(data_dir + "/processed/PPG/ppg/X_test.pkl", "rb") as f:
        test_data = pickle.load(f)
    with open(data_dir + "/processed/PPG/ppg/y_test.pkl", "rb") as f:
        test_labels = pickle.load(f)

    data = (train_data, val_data, test_data)
    labels = (train_labels, val_labels, test_labels)

    return data, labels, lambda x: x


def load_Mocap_dataset(data_dir):
    with open(data_dir + "/processed/Mocap/data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(data_dir + "/processed/Mocap/labels.pkl", "rb") as f:
        labels = pickle.load(f)

    label_map = {"jump": 0, "run": 1, "walk": 2}
    labels = jnp.array([label_map[la] for la in labels])
    onehot_labels = jnp.zeros((len(labels), len(jnp.unique(labels))))
    onehot_labels = onehot_labels.at[jnp.arange(len(labels)), labels].set(1)

    # Pad data
    max_len = np.max([len(d) for d in data])
    padded_seqs = []
    for seq in data:
        num_dim = seq.shape[1]
        padded_seq = np.pad(
            seq.reshape(-1, num_dim),
            pad_width=((0, max_len - len(seq)), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        padded_seqs.append(padded_seq)
    data = jnp.asarray(np.array(padded_seqs))

    bounds = [0.7, 0.85]
    split_data = split(data, bounds)
    split_labels = split(onehot_labels, bounds)

    return split_data, split_labels, lambda x: x


def load_SE3_dataset(name, data_dir):
    with open(data_dir + f"/processed/SE3/{name}/data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(data_dir + f"/processed/SE3/{name}/labels.pkl", "rb") as f:
        labels = pickle.load(f)

    bounds = [0.7, 0.85]
    split_data = split(data, bounds)
    split_labels = split(labels, bounds)

    return split_data, split_labels, lambda x: x


def load_Cifar10_dataset():
    try:
        import torchvision
    except:
        raise RuntimeError("Must have torch/torchvision installed to load Cifar10 dataset.")

    # Load CIFAR-10
    download_dir = "data/raw/cifar"
    dataset_train = torchvision.datasets.CIFAR10(
        download_dir,
        train=True,
        download=True,
        transform=torchvision.transforms.Grayscale(),
    )
    dataset_test = torchvision.datasets.CIFAR10(
        download_dir,
        train=False,
        transform=torchvision.transforms.Grayscale(),
    )
    data_dim = 1  # One grayscale channel
    num_classes = 10

    # CIFAR-10 grayscale normalization (from S5)
    mean = 122.6 / 255.0
    std = 61.0 / 255.0

    # Convert to numpy arrays first (need to do this for tensorflow datasets)
    train_data = []
    train_labels = []
    for image, label in dataset_train:
        train_data.append(np.array(image))
        train_labels.append(np.array(label))
    train_data = jnp.array(train_data).reshape(-1, 32 * 32, data_dim)
    train_data = (train_data / 255 - mean) / std
    train_labels = jax.nn.one_hot(jnp.array(train_labels), num_classes)

    test_data = []
    test_labels = []
    for image, label in dataset_test:
        test_data.append(np.array(image))
        test_labels.append(np.array(label))
    test_data = jnp.array(test_data).reshape(-1, 32 * 32, data_dim)
    test_data = (test_data / 255 - mean) / std
    test_labels = jax.nn.one_hot(jnp.array(test_labels), num_classes)

    bounds = [0.9]  # From S5
    (train_data, val_data) = split(train_data, bounds)
    (train_labels, val_labels) = split(train_labels, bounds)
    data = (train_data, val_data, test_data)
    labels = (train_labels, val_labels, test_labels)

    return data, labels, lambda x: x


def load_NoisyCifar10_dataset():
    try:
        import torchvision
    except:
        raise RuntimeError("Must have torch/torchvision installed to load NoisyCifar10 dataset.")

    # Load CIFAR-10
    download_dir = "data/raw/cifar"
    dataset_train = torchvision.datasets.CIFAR10(
        download_dir,
        train=True,
        download=True,
    )
    dataset_test = torchvision.datasets.CIFAR10(
        download_dir,
        train=False,
        download=True,
    )
    num_classes = 10

    # Convert to numpy arrays first (need to do this for tensorflow datasets)
    train_data = []
    train_labels = []
    for image, label in dataset_train:
        train_data.append(np.array(image))
        train_labels.append(np.array(label))
    train_data = jnp.array(train_data)
    train_labels = jnp.array(train_labels)
    test_data = []
    test_labels = []
    for image, label in dataset_test:
        test_data.append(np.array(image))
        test_labels.append(np.array(label))
    test_data = jnp.array(test_data)
    test_labels = jnp.array(test_labels)

    # Normalize by channel
    mean = np.mean(train_data, axis=[0, 1, 2])
    std = np.std(train_data, axis=[0, 1, 2])
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    # Flatten channels
    train_data = jnp.array(train_data).reshape(-1, 32, 96)
    test_data = jnp.array(test_data).reshape(-1, 32, 96)

    # One-hot labels
    train_labels = jax.nn.one_hot(jnp.array(train_labels), num_classes)
    test_labels = jax.nn.one_hot(jnp.array(test_labels), num_classes)

    # Split data
    bounds = [0.9]  # From S5
    (train_data, val_data) = split(train_data, bounds)
    (train_labels, val_labels) = split(train_labels, bounds)
    data = (train_data, val_data, test_data)
    labels = (train_labels, val_labels, test_labels)

    def data_out_func(self, batch):
        """Noisify during runtime"""
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, shape=(batch.shape[0], 968, batch.shape[-1]))
        noisy_batch = jnp.concatenate([batch, noise], axis=1)
        return noisy_batch

    return data, labels, data_out_func


def create_SequentialCifar10_dataset():
    """
    Flattened RGB Cifar
    """
    try:
        import torchvision
    except:
        raise RuntimeError("Must have torch/torchvision installed to load Cifar10 dataset.")

    # Transform: ToTensor + Normalize + Reshape to (1024, 3)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # (32,32,3) -> (3,32,32) and /255
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        torchvision.transforms.Lambda(lambda x: x.view(3, 1024).t())  # (3,32,32) -> (1024,3)
    ])

    download_dir = "data/raw/sequential_cifar"
    trainset = torchvision.datasets.CIFAR10(
        root=download_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=download_dir, train=False, download=True, transform=transform)

    # Apply transforms by iterating through dataset
    x_train = []
    y_train = []
    for i in range(len(trainset)):
        img, label = trainset[i]  # This applies the transform!
        x_train.append(img.numpy())
        y_train.append(label)
    
    x_test = []
    y_test = []
    for i in range(len(testset)):
        img, label = testset[i]  # This applies the transform!
        x_test.append(img.numpy())
        y_test.append(label)

    # Convert lists to arrays
    x_train = np.array(x_train)  # Shape: (50000, 1024, 3)
    y_train = np.array(y_train)
    x_test = np.array(x_test)    # Shape: (10000, 1024, 3)
    y_test = np.array(y_test)

    # Convert to JAX arrays
    x_train = jnp.array(x_train)
    x_test = jnp.array(x_test)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)
    
    # Create one-hot labels
    train_onehot = jnp.zeros((len(y_train), 10))
    train_onehot = train_onehot.at[jnp.arange(len(y_train)), y_train].set(1)
    test_onehot = jnp.zeros((len(y_test), 10))
    test_onehot = test_onehot.at[jnp.arange(len(y_test)), y_test].set(1)

    bounds = [0.9]  # From S5
    (x_train, x_val) = split(x_train, bounds)
    (train_onehot, val_onehot) = split(train_onehot, bounds)
    data = (x_train, x_val, x_test)
    labels = (train_onehot, val_onehot, test_onehot)

    return data, labels, lambda x: x


def load_IMDb_dataset():
    try:
        import tensorflow as tf
        import string
    except:
        raise RuntimeError("Must have tensorflow installed to load IMDb dataset.")

    start_char = 1
    oov_char = 2
    end_char = 3
    index_from = 4
    max_length = 4096  # per S5
    min_freq = 15  # per S5
    num_class = 2  # positive, negative

    imdb_data_dir = "data/raw/imdb"
    if imdb_data_dir.exists():
        # TODO load from cache
        raise NotImplementedError()
        return

    # File is cached at ~/.keras/dataset/, copied to data/raw/imdb
    # TODO copy/save to dataset_path
    (train_data, train_labels), (test_data, test_labels) = (
        tf.keras.datasets.imdb.load_data(
            start_char=start_char,
            oov_char=oov_char,
            index_from=index_from,
            seed=42,
        )
    )

    # https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb/get_word_index
    idx_counts = Counter()
    for idxs in train_data:
        idx_counts.update(idxs)
    word_index = tf.keras.datasets.imdb.get_word_index()
    inverted_word_index = {
        i + index_from: word
        for (word, i) in word_index.items()
        if idx_counts[i + index_from] >= min_freq
    }

    char_vocab = list(string.printable)
    num_chars = len(char_vocab) + 4  # bos, eos, unk, pad
    char_index = {char: i + index_from for i, char in enumerate(char_vocab)}
    # inverted_char_index = {i: char for char, i in char_index.items()}

    # From word tokens to char tokens
    def convert_tokens(data):
        out = []
        for x in data:
            tokens = []  # Already begins with start token

            for i, idx in enumerate(x):
                # Add a space between words
                if i > 1:
                    tokens.append(char_index[" "])

                # Process word
                if idx in inverted_word_index:
                    chars = list(inverted_word_index[idx])
                    if all([c in char_index for c in chars]):
                        tokens += [char_index[c] for c in chars]
                    else:
                        tokens.append(oov_char)
                elif idx in [0, 1, 2, 3]:
                    tokens.append(idx)
                else:
                    tokens.append(oov_char)

            # Truncate sequence
            if len(tokens) > max_length + 1:
                tokens = tokens[: max_length + 1]

            tokens.append(end_char)
            out.append(np.array(tokens, dtype=int))

        return out

    train_data = convert_tokens(train_data)
    test_data = convert_tokens(test_data)

    train_labels = jax.nn.one_hot(
        np.array(train_labels), num_classes=num_class
    )
    test_labels = jax.nn.one_hot(np.array(test_labels), num_classes=num_class)

    # No validation provided: train = test = 25000
    # Use last 10000 of training set for validation
    bounds = [0.6]
    (train_data, val_data) = split(train_data, bounds)
    (train_labels, val_labels) = split(train_labels, bounds)
    data = (train_data, val_data, test_data)
    labels = (train_labels, val_labels, test_labels)

    def data_out_func(self, batch):
        """One-hot during runtime (dataset too large)"""
        batch_one_hot = jax.nn.one_hot(
            batch.reshape((len(batch), -1)), num_classes=self.num_chars
        )
        return batch_one_hot

    return data, labels, data_out_func, num_chars, num_class


def load_MNIST_dataset():
    try:
        import torchvision
    except:
        raise RuntimeError("Must have torch/torchvision installed to load MNIST dataset.")

    download_dir = "data/raw/mnist"
    dataset_train = torchvision.datasets.MNIST(
        download_dir,
        train=True,
        download=True,
    )
    dataset_test = torchvision.datasets.MNIST(
        download_dir,
        train=False,
    )
    data_dim = 28
    num_classes = 10

    train_data = []
    train_labels = []
    for image, label in dataset_train:
        train_data.append(np.array(image))
        train_labels.append(np.array(label))
    train_data = jnp.array(train_data).reshape(-1, 28, data_dim)
    train_labels = jax.nn.one_hot(jnp.array(train_labels), num_classes)

    # Normalize
    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std

    test_data = []
    test_labels = []
    for image, label in dataset_test:
        test_data.append(np.array(image))
        test_labels.append(np.array(label))
    test_data = jnp.array(test_data).reshape(-1, 28, data_dim)
    test_data = (test_data - mean) / std
    test_labels = jax.nn.one_hot(jnp.array(test_labels), num_classes)

    bounds = [0.9] 
    (train_data, val_data) = split(train_data, bounds)
    (train_labels, val_labels) = split(train_labels, bounds)
    data = (train_data, val_data, test_data)
    labels = (train_labels, val_labels, test_labels)

    return data, labels, lambda x: x
    

def load_sMNIST_dataset():
    try:
        import torchvision
    except:
        raise RuntimeError("Must have torch/torchvision installed to load sMNIST dataset.")

    download_dir = "data/raw/mnist"
    dataset_train = torchvision.datasets.MNIST(
        download_dir,
        train=True,
        download=True,
    )
    dataset_test = torchvision.datasets.MNIST(
        download_dir,
        train=False,
    )
    data_dim = 1
    num_classes = 10

    train_data = []
    train_labels = []
    for image, label in dataset_train:
        train_data.append(np.array(image))
        train_labels.append(np.array(label))
    train_data = jnp.array(train_data).reshape(len(train_data), 28*28, data_dim)
    train_labels = jax.nn.one_hot(jnp.array(train_labels), num_classes)

    # Normalize
    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std

    test_data = []
    test_labels = []
    for image, label in dataset_test:
        test_data.append(np.array(image))
        test_labels.append(np.array(label))
    test_data = jnp.array(test_data).reshape(len(test_data), 28*28, data_dim)
    test_data = (test_data - mean) / std
    test_labels = jax.nn.one_hot(jnp.array(test_labels), num_classes)

    bounds = [0.9] 
    (train_data, val_data) = split(train_data, bounds)
    (train_labels, val_labels) = split(train_labels, bounds)
    data = (train_data, val_data, test_data)
    labels = (train_labels, val_labels, test_labels)

    return data, labels, lambda x: x


def load_Adding500_dataset():
    sql_train = 500
    sql_val = 500
    sql_test = 2000
    size_train = 7000
    size_val = 1500
    size_test = 1500

    train_key = jax.random.PRNGKey(0)
    val_key = jax.random.PRNGKey(1)
    test_key = jax.random.PRNGKey(2)

    def generate_batch(bsz, sql, key):
        """
        data: (bsz, sql, 2)
        labels: (bsz, 1, 1)
        """
        key1, key2, key3 = jr.split(key, 3)
        values = jr.uniform(key1, shape=(bsz, sql, 1))
        half = sql // 2
        half_1 = jr.randint(key2, (bsz,), 0, half)
        half_2 = jr.randint(key3, (bsz,), half, sql)
        def set_indices(idx1, idx2):
            arr = jnp.zeros((sql,))
            arr = arr.at[idx1].set(1)
            arr = arr.at[idx2].set(1)
            return arr
        indices_1d = jax.vmap(set_indices)(half_1, half_2)
        indices = jnp.expand_dims(indices_1d, axis=-1)  # shape: (bsz, sql, 1)
        data = jnp.concatenate((values, indices), axis=2)
        labels = (values * indices).sum(axis=1, keepdims=True)
        return data, labels

    train_data, train_labels = generate_batch(size_train, sql_train, train_key)
    val_data, val_labels = generate_batch(size_val, sql_val, val_key)
    test_data, test_labels = generate_batch(size_test, sql_test, test_key)

    data = (train_data, val_data, test_data)
    labels = (train_labels, val_labels, test_labels)

    return data, labels, lambda x: x


def load_Adding2000_dataset():
    sql_train = 2000
    sql_val = 2000
    sql_test = 2000
    size_train = 7000
    size_val = 1500
    size_test = 1500

    train_key = jax.random.PRNGKey(0)
    val_key = jax.random.PRNGKey(1)
    test_key = jax.random.PRNGKey(2)

    def generate_batch(bsz, sql, key):
        """
        data: (bsz, sql, 2)
        labels: (bsz, 1, 1)
        """
        key1, key2, key3 = jr.split(key, 3)
        values = jr.uniform(key1, shape=(bsz, sql, 1))
        half = sql // 2
        half_1 = jr.randint(key2, (bsz,), 0, half)
        half_2 = jr.randint(key3, (bsz,), half, sql)
        def set_indices(idx1, idx2):
            arr = jnp.zeros((sql,))
            arr = arr.at[idx1].set(1)
            arr = arr.at[idx2].set(1)
            return arr
        indices_1d = jax.vmap(set_indices)(half_1, half_2)
        indices = jnp.expand_dims(indices_1d, axis=-1)  # shape: (bsz, sql, 1)
        data = jnp.concatenate((values, indices), axis=2)
        labels = (values * indices).sum(axis=1, keepdims=True)
        return data, labels

    train_data, train_labels = generate_batch(size_train, sql_train, train_key)
    val_data, val_labels = generate_batch(size_val, sql_val, val_key)
    test_data, test_labels = generate_batch(size_test, sql_test, test_key)

    data = (train_data, val_data, test_data)
    labels = (train_labels, val_labels, test_labels)

    return data, labels, lambda x: x


def load_Adding5000_dataset():
    sql_train = 5000
    sql_val = 5000
    sql_test = 5000
    size_train = 7000
    size_val = 1500
    size_test = 1500

    train_key = jax.random.PRNGKey(0)
    val_key = jax.random.PRNGKey(1)
    test_key = jax.random.PRNGKey(2)

    def generate_batch(bsz, sql, key):
        """
        data: (bsz, sql, 2)
        labels: (bsz, 1, 1)
        """
        key1, key2, key3 = jr.split(key, 3)
        values = jr.uniform(key1, shape=(bsz, sql, 1))
        half = sql // 2
        half_1 = jr.randint(key2, (bsz,), 0, half)
        half_2 = jr.randint(key3, (bsz,), half, sql)
        def set_indices(idx1, idx2):
            arr = jnp.zeros((sql,))
            arr = arr.at[idx1].set(1)
            arr = arr.at[idx2].set(1)
            return arr
        indices_1d = jax.vmap(set_indices)(half_1, half_2)
        indices = jnp.expand_dims(indices_1d, axis=-1)  # shape: (bsz, sql, 1)
        data = jnp.concatenate((values, indices), axis=2)
        labels = (values * indices).sum(axis=1, keepdims=True)
        return data, labels

    train_data, train_labels = generate_batch(size_train, sql_train, train_key)
    val_data, val_labels = generate_batch(size_val, sql_val, val_key)
    test_data, test_labels = generate_batch(size_test, sql_test, test_key)

    data = (train_data, val_data, test_data)
    labels = (train_labels, val_labels, test_labels)

    return data, labels, lambda x: x




# def _generate_selective_copy_batch(
#     bsz,
#     L,
#     M,
#     A,
#     key,
#     n_distractors=0,
#     hard_negatives=False,
#     variable_length=False,
#     min_M=4,
#     marker_repeat=None,
#     distractor_near_end=False,
#     ):
#     """
#     Harder selective copying task.

#     Input:
#       - first prefix region has true memory tokens scattered randomly
#       - optional distractors, including hard negatives
#       - final marker block indicates recall region

#     Output:
#       - the true memory tokens in the order they appeared
#       - padded to length M if variable_length=True
#       - labels shape: (bsz, M, A)

#     Token semantics:
#       0       = blank/noise
#       1..A-2  = normal tokens
#       A-1     = marker token
#     """
#     key_tok, key_pos, key_aux = jr.split(key, 3)

#     if marker_repeat is None:
#         marker_repeat = M

#     # choose per-example memory length
#     if variable_length:
#         key_len, key_tok = jr.split(key_tok)
#         mem_lengths = jr.randint(key_len, shape=(bsz,), minval=min_M, maxval=M + 1)
#     else:
#         mem_lengths = jnp.full((bsz,), M)

#     # sample full-size token bank, later mask unused
#     tokens_full = jr.randint(key_tok, shape=(bsz, M), minval=1, maxval=A - 1)

#     pos_keys = jr.split(key_pos, bsz)
#     aux_keys = jr.split(key_aux, bsz)

#     def build_one_example(tokens_row, m_len, k_pos, k_aux):
#         prefix_len = L + M
#         x_prefix = jnp.zeros((prefix_len,), dtype=jnp.int32)

#         # choose true memory positions
#         perm = jr.permutation(k_pos, prefix_len)
#         mem_pos = jnp.sort(perm[:m_len])

#         true_tokens = tokens_row[:m_len]
#         x_prefix = x_prefix.at[mem_pos].set(true_tokens)

#         # add distractors
#         if n_distractors > 0:
#             k_d1, k_d2 = jr.split(k_aux)

#             occupied = jnp.zeros((prefix_len,), dtype=bool).at[mem_pos].set(True)
#             avail = jnp.arange(prefix_len)[~occupied]

#             if distractor_near_end:
#                 # bias distractors toward the later part of prefix
#                 avail = avail[avail >= max(0, prefix_len // 2)]

#             num_avail = avail.shape[0]
#             nd = jnp.minimum(n_distractors, num_avail)
#             dperm = jr.permutation(k_d1, num_avail)
#             dist_pos = jnp.sort(avail[dperm[:nd]])

#             if hard_negatives:
#                 # reuse true memory token identities as distractors
#                 rep_perm = jr.permutation(k_d2, m_len)
#                 base = true_tokens[rep_perm]
#                 reps = jnp.tile(base, (nd + m_len - 1) // m_len)[:nd]
#                 dist_tokens = reps
#             else:
#                 dist_tokens = jr.randint(k_d2, shape=(nd,), minval=1, maxval=A - 1)

#             x_prefix = x_prefix.at[dist_pos].set(dist_tokens)

#         # final markers
#         markers = jnp.full((marker_repeat,), A - 1, dtype=jnp.int32)
#         x_int = jnp.concatenate([x_prefix, markers], axis=0)

#         # padded targets
#         y_int = jnp.zeros((M,), dtype=jnp.int32)
#         y_int = y_int.at[:m_len].set(true_tokens)

#         return x_int, y_int

#     x_int, y_int = jax.vmap(build_one_example)(tokens_full, mem_lengths, pos_keys, aux_keys)

#     x = jax.nn.one_hot(x_int, A).astype(jnp.float32)   # (B, T, A)
#     y = jax.nn.one_hot(y_int, A).astype(jnp.float32)   # (B, M, A)
#     return x, y




def _generate_selective_copy_batch(
    bsz,
    L,
    M,
    A,
    key,
    n_distractors=0,
    hard_negatives=False,
    variable_length=False,
    min_M=4,
    marker_repeat=None,
    distractor_near_end=False,
    ):
    """
    JAX-safe selective copy generator.

    Input:
      - prefix of length (L + M)
      - true memory tokens are scattered at sparse positions
      - optional distractors are inserted at other positions
      - final marker block indicates recall region

    Output:
      - x: (B, T, A) one-hot
      - y: (B, M, A) one-hot
        If variable_length=True, inactive target positions are padded with token 0.

    Token semantics:
      0       = blank / pad
      1..A-2  = ordinary tokens
      A-1     = marker token
    """
    if marker_repeat is None:
        marker_repeat = M

    prefix_len = L + M

    key_tok, key_pos, key_aux, key_len = jr.split(key, 4)

    # full token bank, always length M
    tokens_full = jr.randint(key_tok, shape=(bsz, M), minval=1, maxval=A - 1)

    # choose memory lengths
    if variable_length:
        mem_lengths = jr.randint(key_len, shape=(bsz,), minval=min_M, maxval=M + 1)
    else:
        mem_lengths = jnp.full((bsz,), M)

    pos_keys = jr.split(key_pos, bsz)
    aux_keys = jr.split(key_aux, bsz)

    def build_one_example(tokens_row, m_len, k_pos, k_aux):
        active_mask = (jnp.arange(M) < m_len)  # (M,)

        # choose M candidate memory positions; only first m_len are active
        perm = jr.permutation(k_pos, prefix_len)
        mem_pos_all = jnp.sort(perm[:M])       # static slice, safe

        true_tokens = jnp.where(active_mask, tokens_row, 0).astype(jnp.int32)

        # scatter only active memory positions
        x_prefix = jnp.zeros((prefix_len,), dtype=jnp.int32)
        scatter_vals = jnp.where(active_mask, tokens_row, 0)
        x_prefix = x_prefix.at[mem_pos_all].set(scatter_vals)

        if n_distractors > 0:
            k_d1, k_d2 = jr.split(k_aux)

            # occupied positions = active memory positions only
            occupied = jnp.zeros((prefix_len,), dtype=bool)
            occupied = occupied.at[mem_pos_all].set(active_mask)

            # build a score vector so we can select allowed distractor positions
            base_scores = jr.uniform(k_d1, shape=(prefix_len,))

            # forbid occupied positions by sending score very negative
            scores = jnp.where(occupied, -1e9, base_scores)

            if distractor_near_end:
                near_end_mask = (jnp.arange(prefix_len) >= (prefix_len // 2))
                scores = jnp.where(near_end_mask, scores, -1e9)

            nd = min(n_distractors, prefix_len)

            # take top-k allowed positions; invalid positions remain at end with -1e9
            _, dist_pos = jax.lax.top_k(scores, nd)
            dist_pos = jnp.sort(dist_pos)

            if hard_negatives:
                # choose distractor tokens by reusing active true tokens
                rep_perm = jr.permutation(k_d2, M)
                base = tokens_row[rep_perm]
                reps = jnp.tile(base, (nd + M - 1) // M)[:nd]
                dist_tokens = reps
            else:
                dist_tokens = jr.randint(k_d2, shape=(nd,), minval=1, maxval=A - 1)

            # mask out invalid distractor positions (those with score < 0)
            valid_dist = (scores[dist_pos] > -1e8)
            dist_tokens = jnp.where(valid_dist, dist_tokens, 0)

            x_prefix = x_prefix.at[dist_pos].set(dist_tokens)

        markers = jnp.full((marker_repeat,), A - 1, dtype=jnp.int32)
        x_int = jnp.concatenate([x_prefix, markers], axis=0)
        y_int = true_tokens

        return x_int, y_int

    x_int, y_int = jax.vmap(build_one_example)(tokens_full, mem_lengths, pos_keys, aux_keys)

    x = jax.nn.one_hot(x_int, A).astype(jnp.float32)   # (B, T, A)
    y = jax.nn.one_hot(y_int, A).astype(jnp.float32)   # (B, M, A)
    return x, y


def load_SelectiveCopy_dataset(
    L=100,
    M=10,
    A=8,
    n_distractors=0,
    hard_negatives=False,
    variable_length=False,
    min_M=4,
    marker_repeat=None,
    distractor_near_end=False,
):
    size_train = 10000
    size_val = 2000
    size_test = 2000

    train_key = jax.random.PRNGKey(0)
    val_key = jax.random.PRNGKey(1)
    test_key = jax.random.PRNGKey(2)

    train_data, train_labels = _generate_selective_copy_batch(
        size_train,
        L,
        M,
        A,
        train_key,
        n_distractors=n_distractors,
        hard_negatives=hard_negatives,
        variable_length=variable_length,
        min_M=min_M,
        marker_repeat=marker_repeat,
        distractor_near_end=distractor_near_end,
    )

    val_data, val_labels = _generate_selective_copy_batch(
        size_val,
        L,
        M,
        A,
        val_key,
        n_distractors=n_distractors,
        hard_negatives=hard_negatives,
        variable_length=variable_length,
        min_M=min_M,
        marker_repeat=marker_repeat,
        distractor_near_end=distractor_near_end,
    )

    test_data, test_labels = _generate_selective_copy_batch(
        size_test,
        L,
        M,
        A,
        test_key,
        n_distractors=n_distractors,
        hard_negatives=hard_negatives,
        variable_length=variable_length,
        min_M=min_M,
        marker_repeat=marker_repeat,
        distractor_near_end=distractor_near_end,
    )

    data = (train_data, val_data, test_data)
    labels = (train_labels, val_labels, test_labels)
    return data, labels, lambda x: x






def _generate_induction_batch(bsz, seq_len, n_vocab, prefix_len, key):
    """
    Generate induction-head examples of the form

      [random filler] [S, M] [random filler] [S]

    where:
      - S is the special induction token
      - M is the token to be recalled at the end

    Input:
      x: (bsz, seq_len, n_vocab + 1) one-hot
    Target:
      y: (bsz, n_vocab) one-hot
    """
    special_tok = n_vocab
    k_mem, k_pos, k_fill = jr.split(key, 3)

    # memory token M in {0, ..., n_vocab-1}
    mem_tok = jr.randint(k_mem, shape=(bsz,), minval=0, maxval=n_vocab)

    # first occurrence position of S inside early prefix
    first_pos = jr.randint(k_pos, shape=(bsz,), minval=0, maxval=prefix_len)

    # Build filler tokens over normal vocab only
    fill_keys = jr.split(k_fill, bsz)

    def build_one_example(m_tok, p0, k):
        # sequence length fixed = seq_len
        # last token will be the second S
        seq = jr.randint(k, shape=(seq_len,), minval=0, maxval=n_vocab)

        # Put first [S, M]
        seq = seq.at[p0].set(special_tok)
        seq = seq.at[p0 + 1].set(m_tok)

        # Put final S
        seq = seq.at[seq_len - 1].set(special_tok)

        # Optional: avoid accidental special token elsewhere
        # already satisfied because filler sampled only from 0..n_vocab-1
        return seq

    x_int = jax.vmap(build_one_example)(mem_tok, first_pos, fill_keys)   # (bsz, seq_len)
    y_int = mem_tok                                                       # (bsz,)

    x = jax.nn.one_hot(x_int, n_vocab + 1).astype(jnp.float32)
    y = jax.nn.one_hot(y_int, n_vocab).astype(jnp.float32)
    return x, y

def load_InductionHead_dataset(seq_len=256, n_vocab=16, prefix_len=10):
    """
    Mamba-style induction-head task:
      train on fixed sequence length 256, vocab size 16, special token added.
    """
    size_train = 10000
    size_val = 2000
    size_test = 2000

    train_key = jax.random.PRNGKey(10)
    val_key = jax.random.PRNGKey(11)
    test_key = jax.random.PRNGKey(12)

    train_data, train_labels = _generate_induction_batch(size_train, seq_len, n_vocab, prefix_len, train_key)
    val_data, val_labels = _generate_induction_batch(size_val, seq_len, n_vocab, prefix_len, val_key)
    test_data, test_labels = _generate_induction_batch(size_test, seq_len, n_vocab, prefix_len, test_key)

    data = (train_data, val_data, test_data)
    labels = (train_labels, val_labels, test_labels)
    return data, labels, lambda x: x
    






def _generate_mqar_batch(bsz, seq_len, n_vocab, n_pairs, key):
    """
    MQAR-style associative recall.

    Construction:
      prefix contains n_pairs distinct (key, value) pairs:
         K1, V1, K2, V2, ..., Kn, Vn
      middle contains filler tokens
      final token is one queried key Kq
      target is corresponding value Vq

    Input:
      x: (bsz, seq_len, n_vocab) one-hot
    Target:
      y: (bsz, n_vocab) one-hot
    """
    k_pairs, k_query, k_fill = jr.split(key, 3)

    pair_keys = jr.split(k_pairs, bsz)
    query_keys = jr.split(k_query, bsz)
    fill_keys = jr.split(k_fill, bsz)

    def build_one_example(kp, kq, kf):
        # sample 2*n_pairs distinct tokens, first half keys, second half values
        perm = jr.permutation(kp, n_vocab)
        keys = perm[:n_pairs]
        vals = perm[n_pairs:2 * n_pairs]

        prefix = jnp.zeros((2 * n_pairs,), dtype=jnp.int32)
        prefix = prefix.at[0::2].set(keys)
        prefix = prefix.at[1::2].set(vals)

        remaining = seq_len - (2 * n_pairs) - 1
        filler = jr.randint(kf, shape=(remaining,), minval=0, maxval=n_vocab)

        q_idx = jr.randint(kq, shape=(), minval=0, maxval=n_pairs)
        query_key = keys[q_idx]
        target_val = vals[q_idx]

        seq = jnp.concatenate([prefix, filler, jnp.array([query_key])], axis=0)
        return seq, target_val

    x_int, y_int = jax.vmap(build_one_example)(pair_keys, query_keys, fill_keys)

    x = jax.nn.one_hot(x_int, n_vocab).astype(jnp.float32)
    y = jax.nn.one_hot(y_int, n_vocab).astype(jnp.float32)
    return x, y


def load_MQAR_dataset(seq_len=256, n_vocab=32, n_pairs=8):
    size_train = 10000
    size_val = 2000
    size_test = 2000

    train_key = jax.random.PRNGKey(20)
    val_key = jax.random.PRNGKey(21)
    test_key = jax.random.PRNGKey(22)

    train_data, train_labels = _generate_mqar_batch(size_train, seq_len, n_vocab, n_pairs, train_key)
    val_data, val_labels = _generate_mqar_batch(size_val, seq_len, n_vocab, n_pairs, val_key)
    test_data, test_labels = _generate_mqar_batch(size_test, seq_len, n_vocab, n_pairs, test_key)

    data = (train_data, val_data, test_data)
    labels = (train_labels, val_labels, test_labels)
    return data, labels, lambda x: x





def load_presplit_pickle_dataset(data_dir):
    with open(data_dir / "X_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open(data_dir / "y_train.pkl", "rb") as f:
        y_train = pickle.load(f)
    with open(data_dir / "X_val.pkl", "rb") as f:
        X_val = pickle.load(f)
    with open(data_dir / "y_val.pkl", "rb") as f:
        y_val = pickle.load(f)
    with open(data_dir / "X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open(data_dir / "y_test.pkl", "rb") as f:
        y_test = pickle.load(f)

    data = (jnp.asarray(X_train), jnp.asarray(X_val), jnp.asarray(X_test))
    labels = (jnp.asarray(y_train), jnp.asarray(y_val), jnp.asarray(y_test))
    return data, labels, lambda x: x







# =============================================
# SECTION: Entrypoint function
# =============================================


def create_dataset(
    name: str,
    data_dir: str,
    classification: bool,
    time_duration: float,
    use_presplit: bool,
    in_memory=False,
    key=None,
):
    if name in get_subfolders(os.path.join(data_dir, "processed", "UEA")):
        data, labels, data_out_func = load_UEA_dataset(name, data_dir)
    elif name in get_subfolders(os.path.join(data_dir, "processed", "SE3")):
        data, labels, data_out_func = load_SE3_dataset(name, data_dir)
    elif name == "Mocap":
        data, labels, data_out_func = load_Mocap_dataset(data_dir)
    elif name == "PPG":
        data, labels, data_out_func = load_PPG_dataset(data_dir)
    elif name == "Cifar10":
        data, labels, data_out_func = load_Cifar10_dataset()
    elif name == "NoisyCifar10":
        data, labels, data_out_func = load_NoisyCifar10_dataset()
    elif name == "SequentialCifar10":
        data, labels, data_out_func = create_SequentialCifar10_dataset()
    elif name == "IMDb":
        data, labels, data_out_func, input_dim, output_dim = load_IMDb_dataset()
    elif name == "MNIST":
        data, labels, data_out_func = load_MNIST_dataset()
    elif name == "sMNIST":
        data, labels, data_out_func = load_sMNIST_dataset()
    elif name == "Adding500":
        data, labels, data_out_func = load_Adding500_dataset()
    elif name == "Adding2000":
        data, labels, data_out_func = load_Adding2000_dataset()
    elif name == "Adding5000":
        data, labels, data_out_func = load_Adding5000_dataset()
    
    elif name == "SelectiveCopy":
        data, labels, data_out_func = load_SelectiveCopy_dataset(
            L=100,
            M=10,
            A=8,
            n_distractors=0,
            hard_negatives=False,
            variable_length=False,
        )

    elif name == "SelectiveCopyLong":
        data, labels, data_out_func = load_SelectiveCopy_dataset(
            L=250,
            M=16,
            A=16,
            n_distractors=0,
            hard_negatives=False,
            variable_length=False,
        )

    elif name == "SelectiveCopyDistractors":
        data, labels, data_out_func = load_SelectiveCopy_dataset(
            L=250,
            M=12,
            A=10,
            n_distractors=16,
            hard_negatives=False,
            variable_length=False,
            distractor_near_end=True,
        )

    elif name == "SelectiveCopyHard":
        data, labels, data_out_func = load_SelectiveCopy_dataset(
            L=250,
            M=16,
            A=16,
            n_distractors=32,
            hard_negatives=True,
            variable_length=False,
            distractor_near_end=True,
        )

    elif name == "SelectiveCopyVariable":
        data, labels, data_out_func = load_SelectiveCopy_dataset(
            L=250,
            M=16,
            A=16,
            n_distractors=32,
            hard_negatives=True,
            variable_length=True,
            min_M=4,
            distractor_near_end=True,
        )


    elif name == "MQAR":
        data, labels, data_out_func = load_MQAR_dataset(
            seq_len=256,
            n_vocab=32,
            n_pairs=8,
        )
    elif name == "MQARLong":
        data, labels, data_out_func = load_MQAR_dataset(
            seq_len=512,
            n_vocab=32,
            n_pairs=8,
        )
    elif name == "MQARHard":
        data, labels, data_out_func = load_MQAR_dataset(
            seq_len=512,
            n_vocab=48,
            n_pairs=12,
        )

    elif name == "InductionHeadShort":
        data, labels, data_out_func = load_InductionHead_dataset(seq_len=200, n_vocab=10, prefix_len=8)
    elif name == "InductionHead":
        data, labels, data_out_func = load_InductionHead_dataset(seq_len=256, n_vocab=16, prefix_len=10)
    elif name == "InductionHeadLong":
        data, labels, data_out_func = load_InductionHead_dataset(seq_len=350, n_vocab=14, prefix_len=10)



    elif name == "SyntheticRegressionSwitch":
        data_dir = BASE_DIR / "data" / "processed" / "synthetic_regression_switch"
        data, labels, data_out_func = load_presplit_pickle_dataset(data_dir)

        
    elif name == "WriteHoldReset":
        data_dir = BASE_DIR / "data" / "processed" / "write_hold_reset"
        data, labels, data_out_func = load_presplit_pickle_dataset(data_dir)

    elif name == "SyntheticRegressionTV":
        data_dir = BASE_DIR / "data" / "processed" / "synthetic_regression_tv"
        data, labels, data_out_func = load_presplit_pickle_dataset(data_dir)



    elif name == "WriteHoldEraseQuery":
        data_dir = BASE_DIR / "data" / "processed" / "write_hold_erase_query"
        data, labels, data_out_func = load_presplit_pickle_dataset(data_dir)

    elif name == "ModeSwitchOscillator":
        data_dir = BASE_DIR / "data" / "processed" / "mode_switch_oscillator"
        data, labels, data_out_func = load_presplit_pickle_dataset(data_dir)


    else:
        raise ValueError(f"Unknown dataset: {name}")

    if not use_presplit:
        shuffle_key, key = jr.split(key)
        data, labels = shuffle(data, labels, shuffle_key)

    if time_duration is not None:
        data = append_time(data, time_duration)

    if name == "IMDb":
        dataloader_type = BucketedDataloader
    else:
        dataloader_type = StandardDataloader
        input_dim, output_dim = calculate_dimension(data, labels, classification)

    return Dataset(
        name,
        data,
        labels,
        dataloader_type,
        input_dim,
        output_dim,
        in_memory,
        data_out_func,
    )
