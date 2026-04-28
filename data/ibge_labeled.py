import lightning as L
import torch
import os
import json
import numpy as np
import tifffile as tiff
from pathlib import Path
from einops import rearrange
from scipy.ndimage import generic_filter
from typing import Any, Callable
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import Compose

from .transforms import (
    ToTensor,
    Normalize,
    SampleChannelBands,
    SampleTimestamps,
    RandomFlip,
    RandomRotate,
    RandomContrast,
    RandomCrop
)


class IBGE_Labeled(Dataset):
    def __init__(
            self,
            data_dir: str | Path,
            mode: str,
            fold: int | str,
            img_size: int,
            normalize: bool = True,
            transform: Callable[[dict], Any] | None = None
        ) -> None:
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.fold = str(fold)
        self.img_size = img_size
        self.normalize = normalize
        self.transform = transform
        
        self.metadata = self._load_metadata()
        self.full_images = self._load_full_images()
        self.coords = self._compute_coords()
    
    def __len__(self) -> int:
        return len(self.coords)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_index, top, left = self.coords[idx]
        data, target = self.full_images["data"][img_index], self.full_images["target"][img_index]

        data = data[..., top:top+self.img_size, left:left+self.img_size]
        data = np.nan_to_num(data).astype(np.float32)

        # if self.normalize:
        #     data = self._normalize_data(data)

        target = target[:, top:top+self.img_size, left:left+self.img_size]
        target = self._binarize_labels(target)

        sample = {
            "data": data,
            "target": target,
            "coords": torch.tensor([img_index, top, left], dtype=torch.int32)
        }

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _load_metadata(self) -> dict[str, list]:
        assert self.mode in ["train", "val", "test"], f"Invalid mode {self.mode}. \
                                                        Choose from ['train', 'val', 'test']"
        assert int(self.fold) in range(1, 6), f"Invalid fold {self.fold}. Choose from range [1, 5]"

        if self.mode == "test":
            metadata_file_path = self.data_dir / f"test_metadata.json"
        else:
            metadata_file_path = self.data_dir / f"{self.mode}_metadata_fold_{self.fold}.json"
        
        if not metadata_file_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file_path}")
        
        with open(metadata_file_path) as f:
            metadata = json.load(f)
        
        # Using mean and std values from training data
        # if "mean" not in metadata.keys():
        #     train_metadata_path = self.data_dir / f"train_metadata_fold_{self.fold}.json"

        #     with open(train_metadata_path) as f:
        #         train_metadata = json.load(f)
        #         metadata["mean"] = train_metadata["mean"]
        #         metadata["std"] = train_metadata["std"]
        
        return metadata

    def _load_full_images(self) -> dict[str, Any]:
        all_data, all_targets = [], []

        for city_id in self.metadata["city_ids"]:
            data_paths = sorted([f for f in os.listdir(self.data_dir / "sentinel-2") if f.startswith(city_id)])
            data = np.stack([tiff.imread(self.data_dir / "sentinel-2" / f) for f in data_paths], axis=0)
            data = rearrange(data, "t h w c -> t c h w")
            all_data.append(data)

            target = tiff.imread(self.data_dir / "annotated_parcels" / f"{city_id}_reference_instance.tif")
            target = rearrange(target, "h w -> 1 h w")
            all_targets.append(target)
        
        return {
            "data": all_data,
            "target": all_targets
        }
        
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        # Converting to reflectance
        data = data / 10_000

        mean = np.array(self.metadata["mean"]).astype(np.float32)
        std = np.array(self.metadata["std"]).astype(np.float32)
        
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]

        return (data - mean) / std
    
    def _binarize_labels(self, target: np.ndarray) -> np.ndarray:
        return np.where(target > 0, 1, 0).astype(np.float32)
    
    # def _process_labels(self, target: np.ndarray) -> np.ndarray:
    #     def compute_margins(target):
    #         return len(np.unique(target)) > 1
        
    #     # Binarizing mask for background vs crop
    #     bin_mask = (target > 0).astype(int)
    #     bin_mask[bin_mask > 0] = 1
        
    #     mrg_size = 5

    #     margin = generic_filter(target, compute_margins, size=[1, mrg_size, mrg_size])
    #     bin_mask[margin > 0] = 2
        
    #     return bin_mask

    def _compute_coords(self) -> list[tuple[int, int, int]]:
        all_data = self.full_images["data"]
        coords = []

        # stride = int(self.img_size * 0.5) if self.mode == "test" else self.img_size
        stride = self.img_size

        for img_index, data in enumerate(all_data):
            H, W = data.shape[-2:]

            for top in range(0, H - self.img_size + 1, stride):
                for left in range(0, W - self.img_size + 1, stride):
                    patch = data[..., top:top+self.img_size, left:left+self.img_size]

                    # Skipping all-zero and all-NaN patches
                    if np.all(patch == 0) or np.isnan(patch).all():
                        continue

                    coords.append((img_index, top, left))

        return coords


class IBGE_Labeled_Module(L.LightningDataModule):
    def __init__(self, config: Any) -> None:
        super().__init__()

        self.validate = config.dataset.get("validate", False)
        self.num_workers = config.dataset.get("num_workers", 4)

        self.batch_size = config.dataset.batch_size
        self.num_channels = config.model.num_channels
        self.num_timestamps = config.model.num_timestamps
        self.patch_size = config.model.patch_size
        self.crop_size = config.model.crop_size
        self.fold = config.dataset.fold
        self.data_dir = Path(config.dataset.data_dir)

        if self.num_channels == 3:
            channel_bands = [3, 2, 1]
        elif self.num_channels == 4:
            channel_bands = [1, 2, 3, 7]
        else:
            raise ValueError("Number of channels must be either 3 (RGB) or 4 (BGRNIR)")

        self.transform_train = Compose([
            Normalize(channel_bands=channel_bands),
            # SampleChannelBands(channel_bands=channel_bands),
            SampleTimestamps(num_timestamps=self.num_timestamps, sample_type="random"),
            RandomCrop(size=(self.crop_size, self.crop_size)),
            RandomFlip(prob=0.5, orientation="hor"),
            RandomFlip(prob=0.5, orientation="ver"),
            RandomRotate(),
            # RandomContrast(),
            ToTensor()
        ])
        self.transform_test = Compose([
            Normalize(channel_bands=channel_bands),
            # SampleChannelBands(channel_bands=channel_bands),
            SampleTimestamps(num_timestamps=self.num_timestamps, sample_type="first"),
            ToTensor()
        ])

    def setup(self, stage: str | None = None) -> None:
        """
        Setup the dataset for training, validation, and testing.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = IBGE_Labeled(self.data_dir,
                                              mode="train",
                                              fold=self.fold,
                                              img_size=self.patch_size,
                                              transform=self.transform_train)
            if self.validate:
                self.val_dataset = IBGE_Labeled(self.data_dir,
                                                mode="val",
                                                fold=self.fold,
                                                img_size=self.crop_size,
                                                transform=self.transform_test)
        if stage == "test" or stage is None:
            self.test_dataset = IBGE_Labeled(self.data_dir,
                                             mode="test",
                                             fold=self.fold,
                                             img_size=self.crop_size,
                                             transform=self.transform_test)

    def train_dataloader(self) -> Callable[[dict], DataLoader]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True
        )
    
    def val_dataloader(self) -> Callable[[dict], DataLoader] | list:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False
        ) if self.validate else []

    def test_dataloader(self) -> Callable[[dict], DataLoader]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False
        )
