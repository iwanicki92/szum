from pathlib import Path
import shutil
from typing import Any
import numpy as np
from numpy.typing import NDArray

from methods import Label, LabeledDataset, Range, Split

import scipy.ndimage

def augment(data):
    rotations_per_image = 2
    augment_count = len(data)

    rng = np.random.default_rng(seed=666_666)

    label_sizes = {Label.PAPER: augment_count // 3, Label.ROCK: augment_count // 3}
    label_sizes[Label.SCISSORS] = augment_count - sum(label_sizes.values())

    random_images = {
        label: rng.choice(data[label], size=label_sizes[label], replace=False)
        for label in Label
    }

    def rotate(image, angle) -> NDArray:
        return scipy.ndimage.rotate(image, angle, reshape=False, mode="nearest")

    augmented_images = {
        label: [
            rotate(image, angle)
            for image in random_images[label]
            for angle in rng.choice(range(1, 360), size=rotations_per_image)
            ] for label in Label
        }
    
    for label, images in augmented_images.items():
        augmented_images[label] = np.vstack([np.array(images), random_images[label]])

    return LabeledDataset(_labels_with_dataset=augmented_images)

def normalize(splits: list[Split]):
    _, mean, std = splits[1].train.normalize(inplace=True)
    _, _, _ = splits[1].val.normalize(mean=mean, std=std, inplace=True)
    _, _, _ = splits[1].test.normalize(mean=mean, std=std, inplace=True)
    splits[1] = Split(splits[1].train, splits[1].val, splits[1].test)

def create_split3(splits: list[Split]):
    split3_train = LabeledDataset(_labels_with_dataset= {
        label: np.vstack([splits[1].train[label], splits[1].val[label]])
        for label in Label
    })
    splits.append(Split(split3_train, splits[1].val, splits[1].test))

def create_dataset():
    dataset_path = Path("dataset")
    dataset = LabeledDataset(dataset_path=dataset_path)
    dataset_unchanged = LabeledDataset(dataset_path=dataset_path, uniform_size=False, new_size=None)
    splits = [Split(*dataset_unchanged.split_dataset()), Split(*dataset.split_dataset())]
    train = splits[1].train
    print("Augmenting dataset")
    augmented = augment(train)
    splits[1] = Split(augmented.copy(), splits[1].val, splits[1].test)
    print("Normalizing dataset")
    normalize(splits)
    create_split3(splits)
    return splits

def save_dataset(splits: list[Split]):
    splits_path = Path(f"dataset/splits")
    splits_path.mkdir(exist_ok=True)
    for i, split in enumerate(splits):
        split_path = splits_path / f"split_{i+1}"
        split_path.mkdir(exist_ok=True)
        for set, set_dataset in zip(split._fields, split):
            set_path = split_path / f"{set}"
            set_path.mkdir(exist_ok=True)
            for label in Label:
                label_path = set_path / label.name
                np.save(label_path, set_dataset[label])
        if split.train.mean is not None and split.train.std is not None:
            np.save(split_path / 'mean', split.train.mean)
            np.save(split_path / 'std', split.train.std)

def load_dataset(*, recreate = False, save = True):
    """_summary_

    Args:
        recreate (bool, optional):  If true then recreate dataset even if it can be loaded. Defaults to False.
        save (bool, optional): If dataset was recreated then save to disk. Defaults to True.

    Returns:
        _type_: _description_
    """
    if not Path("dataset/splits").is_dir() or recreate:
        print("Creating dataset")
        splits = create_dataset()
        if save:
            if Path("dataset/splits").exists():
                print("Deleting old dataset")
                shutil.rmtree(Path("dataset/splits"))
            print("Saving dataset")
            save_dataset(splits)
        return splits

    print("Loading dataset")
    splits: list[Split] = []
    for split in [f"split_{i}" for i in range(1,4)]:
        sets: dict[str, dict[Label, NDArray]] = {}
        mean, std = None, None
        for set in Split._fields:
            sets[set] = {
                label: np.load(f"dataset/splits/{split}/{set}/{label.name}.npy", allow_pickle=True)
                for label in Label
                }
        mean_path = Path(f"dataset/splits/{split}/mean")
        std_path = Path(f"dataset/splits/{split}/std")
        if mean_path.exists() and std_path.exists():
            mean = np.load(mean, allow_pickle=True)
            std = np.load(std, allow_pickle=True)
        splits.append(Split(**{
            set_str: LabeledDataset(_labels_with_dataset=set, mean=mean, std=std)
            for set_str, set in sets.items()
            }))
    
    return splits