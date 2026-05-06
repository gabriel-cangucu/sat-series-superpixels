import torch
import os
import warnings
import numpy as np
import pandas as pd
from einops import rearrange
from pathlib import Path


class ToTensor:
    def __init__(self, with_datetime: bool = False) -> None:
        self.with_datetime = with_datetime
        
    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """
        Convert the input sample to a tensor.
        """
        data = sample["data"]
        assert len(data.shape) == 4, "Expected a 4D array"
        
        data = rearrange(data, "t c h w -> c t h w")
        sample["data"] = torch.from_numpy(data.copy()).float()
        
        if "target" in sample.keys():
            target = sample["target"]
            assert len(target.shape) == 3, "Expected target array to be 1 x H x W"

            sample["target"] = torch.from_numpy(target.copy()).long()
        
        if self.with_datetime:
            dates = sample["dates"]
            sample["dates"] = torch.from_numpy(dates.copy()).float()
        
        return sample


# class Normalize:
#     def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
#         data = sample["data"]
#         data = data / 10_000.0  # Converting to reflectance
        
#         data_with_nans = np.where(data > 0, data, np.nan)

#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=RuntimeWarning)
#             p99 = np.nanpercentile(data_with_nans, 99, axis=(2, 3), keepdims=True)

#         p99 = np.nan_to_num(p99, nan=1.0)
        
#         # Clipping outliers per band and timestamp
#         data = np.clip(data / (p99 + 1e-6), 0, 1)
#         data = np.where(np.isnan(data_with_nans), 0, data)
        
#         sample["data"] = data
#         return sample


class SampleChannelBands:
    def __init__(self, channel_bands: list[int]) -> None:
        self.channel_bands = channel_bands
    
    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Sample channel bands from the input data.
        """
        data = sample["data"]
        data = data[:, self.channel_bands]
        sample["data"] = data

        return sample


class SampleTimestamps:
    def __init__(
            self,
            num_timestamps: int = 3,
            with_datetime: bool = False,
            sample_type: str = "random"
        ) -> None:
        self.num_timestamps = num_timestamps
        self.with_datetime = with_datetime
        self.sample_type = sample_type

    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Sample timestamps from the input data.
        """
        data = sample["data"]

        if data.shape[0] < self.num_timestamps:
            raise ValueError(f"Number of timestamps in data is less than {self.num_timestamps}.")
        
        if self.sample_type == "random":
            timestamps = np.sort(np.random.choice(data.shape[0], size=self.num_timestamps, replace=False))
        elif self.sample_type == "first":
            timestamps = np.arange(self.num_timestamps)
        else:
            raise ValueError("Sample type must be either 'random' or 'first'.")

        sample["data"] = data[timestamps]
        
        if self.with_datetime:
            dates = sample["dates"]
            sample["dates"] = dates[timestamps]

        return sample


class Normalize:
    def __init__(self, p_low: int = 2, p_high: int = 98, channel_bands: list[int] = [3, 2, 1]) -> None:
        self.p_low = p_low
        self.p_high = p_high
        self.channel_bands = channel_bands

    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Normalize timestamps to 0-1 using percentiles.
        """
        rgb_timestamps = []

        for timestamp in sample["data"]:
            rgb = timestamp[self.channel_bands, :, :]   # (C, H, W)

            lo = np.percentile(rgb, self.p_low, axis=(1, 2), keepdims=True)
            hi = np.percentile(rgb, self.p_high, axis=(1, 2), keepdims=True)

            rgb = np.clip((rgb - lo) / (hi - lo + 1e-5), 0, 1)
            rgb_timestamps.append(rgb)
        
        sample["data"] = np.stack(rgb_timestamps, axis=0)
        return sample
    

class RandomFlip:
    def __init__(self, prob: float = 0.5, orientation: str = "hor") -> None:
        self.prob = prob
        self.orientation = orientation

    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Randomly flip the data horizontally with probability prob.
        """
        if self.orientation == "hor":
            axis = -1
        elif self.orientation == "ver":
            axis = -2
        else:
            raise ValueError("Orientation must be either 'hor' or 'ver'.")
        
        if np.random.rand() < self.prob:
            sample["data"] = np.flip(sample["data"], axis=axis)
            
            if "target" in sample.keys():
                sample["target"] = np.flip(sample["target"], axis=axis)
        
        return sample


class RandomRotate:
    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Rotate images and labels with a random degree in {0, 90, 180, 270}.
        """
        num_rotations = np.random.randint(4)
        
        if num_rotations > 0:
            sample["data"] = np.rot90(sample["data"], k=num_rotations, axes=(-2, -1))
            
            if "target" in sample.keys():
                sample["target"] = np.rot90(sample["target"], k=num_rotations, axes=(-2, -1))
        
        return sample


class RandomCrop:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
    
    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Crop a time series to a new size.
        """
        data = sample["data"]
        assert len(data.shape) == 4, "Expected a 4D array for the data"
        
        _, _, h, w = data.shape
        crop_h, crop_w = self.size
        
        if h < crop_h or w < crop_w:
            raise ValueError("Crop size must be smaller than the image size.")
        
        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)
    
        sample["data"] = data[:, :, top:top+crop_h, left:left+crop_w]
        
        if "target" in sample.keys():
            target = sample["target"]
            assert len(target.shape) == 3, "Expected target array to be 1 x H x W"
            
            sample["target"] = target[:, top:top+crop_h, left:left+crop_w]
        
        return sample


class FilterClouds:
    def __init__(self, data_dir: str | Path, threshold = 0.1, with_datetime : bool = False):
        self.data_dir = data_dir
        self.threshold = threshold
        self.with_datetime = with_datetime

        csv_path = Path(data_dir) / "pastis_cloud_analysis.csv"
        
        if not os.path.isfile(csv_path):
            raise FileNotFoundError("'pastis_cloud_analysis.csv' not found in root data dir.")
        
        self.valid_indices = self._get_valid_indices(csv_path)
    
    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Keep only timestamps whose cloud percentage is below the threshold.
        """
        data, stem = sample["data"], sample["stem"]
        
        indices = self.valid_indices[stem]
        sample["data"] = data[indices]
        
        if self.with_datetime:
            dates = sample["dates"]
            sample["dates"] = dates[indices]
        
        return sample
    
    def _get_valid_indices(self, csv_path: Path) -> dict[str, list]:
        cloud_df = pd.read_csv(csv_path)
        valid_indices = {}
        
        for stem, group in cloud_df.groupby("stem"):
            indices_list = group[group["cloud_percentage"] < self.threshold]["timestamp"].tolist()
            valid_indices[stem] = indices_list
        
        return valid_indices


class RandomContrast:
    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Apply random gamma correction.
        """
        data = sample["data"]

        if np.random.random() < 0.5:
            # Random gamma value
            gamma = 0.75 + (np.random.random() * 0.5)
        
            for c in range(data.shape[1]):
                min_val = data[:, c].min()
                max_val = data[:, c].max()

                data[:, c] = (data[:, c] - min_val) / (max_val - min_val + 1e-5)
                data[:, c] = data[:, c]**gamma
                data[:, c] = (data[:, c] * (max_val - min_val + 1e-5)) + min_val

        sample["data"] = data
        return sample
