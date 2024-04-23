from enum import IntEnum
from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple
import cv2
from cv2.typing import MatLike
import numpy as np
from numpy.typing import NDArray
import skimage.filters as sk_filters


class Channel(IntEnum):
    """OpenCV uses BGR order instead of RGB"""
    B = 0
    G = 1
    R = 2


class Label(IntEnum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2


class Range(NamedTuple):
        start: int
        end: int


class Dataset:
    RNG_SEED = 1234

    def __init__(self, dataset: NDArray[np.uint8] | NDArray[np.float64] | None = None) -> None:
        self.data = np.empty(0) if dataset is None else dataset
        self._rng = np.random.default_rng(seed=self.RNG_SEED)

    def __getitem__(self, key):
        return self.data.__getitem__(key)
    
    def __len__(self):
        return len(self.data)


class LabeledDataset(Dataset):
    def __init__(self, *,
                 dataset_path: PathLike | str = None,
                 uniform_size = True, 
                 new_size: tuple[int, int] | None = (200, 200),
                 mean: Any = None,
                 std: Any = None,
                 **kwargs
                 ) -> None:
        """load dataset

        Args:
            dataset_path (str): directory containing `rock`, `paper` and `scissor` directories with images
            uniform_size (bool, optional): Make sure all labels contain same amount of images by removing excess.
            new_size (tuple[int, int] | None, optional): If not None then resize images to (w, h) pixels. Defaults to (200, 200).
        """
        self.mean = mean
        self.std = std

        if "_dataset" in kwargs:
            data, self._label_ranges = kwargs["_dataset"]
            super().__init__(data)
            return
        
        if "_labels_with_dataset" in kwargs:
            data: dict[Label, NDArray] = kwargs['_labels_with_dataset']
            ranges = np.cumsum([0, *[len(images) for images in data.values()]])
            ranges = [Range(start, end) for start, end in zip(ranges[:], ranges[1:])]
            ranges = {label: range for label, range in zip(Label, ranges)}
            self._label_ranges = ranges
            super().__init__(np.vstack(list(data.values())))
            return
        
        super().__init__()

        labeled_images = self._load_images(dataset_path, new_size)
        if uniform_size:
            labeled_images = self._remove_excess(labeled_images)
        self.data = np.vstack(list(labeled_images.values()))

        # remember indexes (start/end) of labels
        self._label_ranges: dict[Label, Range] = {}
        start_index = 0
        for label, images in labeled_images.items():
            self._label_ranges[label] = Range(start_index, len(images) + start_index)
            start_index += len(images)

    def copy(self):
        return LabeledDataset(_dataset=(self.data.copy(), self._label_ranges.copy()))

    def normalize(self, *, mean: Any | None = None, std: Any | None = None, inplace = False):
        # convert to greyscale & apply scharr (edge detection) filter
        grey_dataset = np.empty(self.data.shape[:-1], dtype=np.float64)
        for i, image in enumerate(self.data):
            grey_dataset[i] = sk_filters.scharr(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        self.data = grey_dataset

        # calculate mean and std for each channel
        mean: np.ndarray = self.data.mean() if mean is None else mean
        std: np.ndarray = self.data.std() if std is None else std
        normalized = (self.data - mean) / std
        if inplace:
            self.data = normalized
            self.mean, self.std = mean, std
            return self, mean, std
        return LabeledDataset(mean=mean, std=std, _dataset=(normalized, self._label_ranges.copy())), mean, std
        
    def __getitem__(self, key):
        if isinstance(key, Label):
            return self.data[self._label_ranges[key].start : self._label_ranges[key].end]
        elif isinstance(key, Channel):
            return self.data[..., key]
        return super().__getitem__(key)

    def _load_images(self, dataset_path: PathLike, resize):
        dirs = {
            Label.ROCK: dataset_path / Path("rock"),
            Label.PAPER: dataset_path / Path("paper"),
            Label.SCISSORS: dataset_path / Path("scissors")
        }
        labeled_images: dict[Label, list[MatLike]] = {}

        for label, dir in dirs.items():
            labeled_images[label] = []
            for img_path in dir.glob("*.png"):
                try:
                    img = cv2.imread(str(img_path))
                    if resize is not None and img.shape != resize:
                        img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
                    labeled_images[label].append(img)
                except Exception as ex:
                    print(f'{img_path}: {ex}')
        return labeled_images

    def _remove_excess(self, labeled_images: dict[Label, list[MatLike]]):
        min_size = len(min(labeled_images.values(), key=len))
        uniform_labeled_images: dict[Label, list[MatLike]] = {}

        for label, images in labeled_images.items():
            size = len(images)
            num_to_remove = size - min_size
            if num_to_remove <= 0:
                uniform_labeled_images[label] = images
                continue
            indexes_to_remove = self._rng.choice(size, size=num_to_remove, replace=False)
            uniform_labeled_images[label] = [image for index, image in enumerate(images) if index not in indexes_to_remove]

        return uniform_labeled_images
    
    def split_dataset(self, train: int = 70, val: int = 20, test: int = -1):
        """Split dataset into training, validation and test datasets.

        Each argument is size as percentage of dataset. Use -1 to calculate remaining percentage,
        e.g. split_dataset(50, -1, 10) == split_dataset(50, 40, 10)

        Args:
            train (int, optional): How big should the training set be. Defaults to 70%.
            val (int, optional): How big should the validation set be. Defaults to 20%.
            test (int, optional): How big should the testing set be. Defaults to -1.
        """
        sizes = [train, val, test]
        if sizes.count(-1) > 1:
            raise Exception("Only one -1 is allowed")
        if -1 in sizes:
            sizes[sizes.index(-1)] = 100 - sum(percent for percent in sizes if percent != -1)
        if min(sizes) < 0 or max(sizes) > 100 or sum(sizes) != 100:
            raise Exception("Values must be between <0,100> or -1")
        
        def split_size(percent, data) -> int:
            return round(len(data) * percent / 100)
        
        # train, val
        sizes = [
            {label: split_size(train, self[label]) for label in Label},
            {label: split_size(val, self[label]) for label in Label}
            ]
        # test (rest of images)
        sizes.append({label: len(self[label]) - sum(split[label] for split in sizes) for label in Label})
        index = {label: 0 for label in Label}
        splits: list[dict[Label, Range]] = []
        new_label_ranges: list[dict[Label, Range]] = []
        for size in sizes:
            end_index = {label: index[label] + size[label] for label in Label}
            splits.append({label: Range(index[label], end_index[label]) for label in Label})
            ranges: dict[Label[Range]] = {}
            start_indices = [0, *np.cumsum([*size.values()])]
            for label, i in zip(Label, range(len(start_indices) - 1)):
                ranges[label] = Range(start_indices[i], start_indices[i+1])
            new_label_ranges.append(ranges)
            index = end_index

        return [
            LabeledDataset(
                _dataset=(
                    np.vstack([self[label][split[label].start : split[label].end] for label in Label]),
                    new_indices
                    )
                )
                for new_indices, split in zip(new_label_ranges, splits)
        ]


class Split(NamedTuple):
    train: LabeledDataset
    val: LabeledDataset
    test: LabeledDataset


def main():
    dataset = LabeledDataset(dataset_path="dataset")
    dataset.split_dataset()
    return


if __name__ == "__main__":
    main()