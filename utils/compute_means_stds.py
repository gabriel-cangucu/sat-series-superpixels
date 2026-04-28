import os
import argparse
import rasterio
import numpy as np
import json
from pathlib import Path


def compute_s2_normalization(tif_paths: list[Path], scale_factor: float = 10000.0, nodata_value: int = 0):
    """
    Computes the channel-wise mean and standard deviation of a Sentinel-2 image,
    ignoring NoData values (zeros and NaNs) and converting to reflectance.
    """
    with rasterio.open(tif_paths[0]) as src:
        num_bands = src.count
    
    pixel_counts = np.zeros(num_bands)
    sum_reflectance = np.zeros(num_bands)
    sum_sq_reflectance = np.zeros(num_bands)

    print(f"Processing {num_bands} bands...")
    print(f"Starting calculation across {len(tif_paths)} images...\n")
    
    for path in tif_paths:
        print(f"Processing: {path.name}")

        with rasterio.open(path) as src:
            num_bands = src.count
            
            # Looping through each band (Rasterio bands are 1-indexed)
            for band_idx in range(1, num_bands + 1):
                band_data = src.read(band_idx)
                
                # Creating a boolean mask for valid pixels
                valid_mask = (band_data != nodata_value) & (~np.isnan(band_data))
                valid_pixels = band_data[valid_mask].astype(np.float32)

                # Converting to reflectance (divide by 10,000)
                valid_pixels /= scale_factor

                if valid_pixels.size > 0:
                    # Clipping (1st and 99th percentile)
                    p1, p99 = np.percentile(valid_pixels, [1, 99])
                    valid_pixels = np.clip(valid_pixels, p1, p99)
                    
                    pixel_counts[band_idx - 1] += valid_pixels.size
                    sum_reflectance[band_idx - 1] += np.sum(valid_pixels)
                    sum_sq_reflectance[band_idx - 1] += np.sum(valid_pixels**2)
                else:
                    print(f"Warning: band {band_idx} from {path.name} has no valid pixels!")

    final_means = sum_reflectance / pixel_counts
    # Variance = (Sum of Squares / N) - Mean^2
    final_vars = (sum_sq_reflectance / pixel_counts) - (final_means**2)
    final_stds = np.sqrt(np.maximum(final_vars, 0)) # Max to avoid tiny negative precision errors

    return final_means.tolist(), final_stds.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes the mean and std for multiple tif files")

    parser.add_argument("--data_path", type=str, help="Data root path")
    parser.add_argument("--city_ids", nargs="+", help="List of city IDS to contribute to the computation")

    args = parser.parse_args()

    tif_file_paths = []

    for file in os.listdir(args.data_path):
        for city_id in args.city_ids:
            if file.startswith(city_id):
                tif_file_paths.append(Path(args.data_path) / file)
    
    # Note: If your specific file uses a different nodata value (e.g., -9999), change it here.
    means, stds = compute_s2_normalization(tif_file_paths, nodata_value=0)
    metadata = {
        "city_ids": args.city_ids,
        "mean": means,
        "std": stds
    }

    with open("means_stds.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    print("\n--- Final Statistics ---")
    print(f"Means = {means}")
    print(f"Stds  = {stds}")
