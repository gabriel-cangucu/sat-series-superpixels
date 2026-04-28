import os
import torch
import torchvision
import random
import omegaconf
import numpy as np
import lightning as L
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.utils import make_grid
from collections import defaultdict
from typing import Callable, Any
from pathlib import Path
from tqdm import tqdm
from einops import rearrange


def get_data_sample(dataset: Dataset, indices: list[int] | int) -> DataLoader:
    """
    Generate a dataloader from a sample of the data. The sample is random unless indices are specified.
    """
    if isinstance(indices, int):
        indices = random.sample(range(len(dataset)), k=indices)
        
    subset = Subset(dataset, indices)
    subset_loader = DataLoader(subset, batch_size=16)
    
    return subset_loader


def normalize_image(image: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    """
    Normalize image to 0-255 for visualization using percentiles.
    """
    lo = np.percentile(image, p_low, axis=(-2, -1), keepdims=True)
    hi = np.percentile(image, p_high, axis=(-2, -1), keepdims=True)

    image = np.clip((image - lo) / (hi - lo + 1e-5), 0, 1)
    
    return (image * 255).astype(np.uint8)


def time_series_to_rgb(time_series: torch.Tensor) -> list[np.ndarray]:
    """
    Convert a time series to RGB images.
    """
    if len(time_series.shape) == 3:
        # Adding a time dimension if images are 2D
        time_series = time_series.unsqueeze(1)

    images = []
    
    for t in range(time_series.shape[1]):
        image = time_series[:, t, :, :].detach().cpu().numpy()
        
        rgb_image = image[[2, 1, 0], :, :] # Indices for B04 (red), B03 (green), B02 (blue)
        rgb_image = np.stack([normalize_image(rgb_image[i]) for i in range(3)], axis=0)
        rgb_image = rearrange(rgb_image, "c h w -> h w c")
        
        images.append(rgb_image)

    return images


def store_preds_as_images(batch: dict[str, torch.Tensor], save_dir: str | Path) -> None:
    """
    Given a batch of time series, store each time series as a grid PNG image
    """
    save_dir = Path(save_dir) / "predictions"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    inputs, preds = batch["inputs"], batch["preds"]
    
    for idx, (image, pred) in tqdm(enumerate(zip(inputs, preds)), total=len(inputs)):
        image = time_series_to_rgb(image)
        pred = time_series_to_rgb(pred)
        
        combined = np.concatenate([image, pred], axis=0)
        grid = make_grid(torch.tensor(combined), nrow=len(image), padding=5, pad_value=255)

        grid_image = torchvision.transforms.ToPILImage()(grid)
        grid_image.save(save_dir / f"{idx}.png")
        

def get_preds_from_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    if len(logits.shape) < 4:
        logits = logits.unsqueeze(0) # [B, C, H, W]

    if num_classes == 1:
        # Binary case: sigmoid + threshold
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
    else:
        # Multiclass case: argmax
        preds = torch.argmax(logits, dim=1)

    return preds


def load_from_checkpoint(
        config: Any,
        model: L.LightningModule,
        model_name: Callable[[dict], L.LightningModule]
    ) -> L.LightningModule:
    if config.checkpoint.ckpt_path is not None:
        torch.serialization.add_safe_globals([
            omegaconf.DictConfig,
            omegaconf.base.ContainerMetadata,
            omegaconf.base.Metadata,
            omegaconf.nodes.AnyNode,
            Any,
            dict,
            defaultdict
        ])

        # Loads ALL model weights
        model = model_name.load_from_checkpoint(
            checkpoint_path=config.checkpoint.ckpt_path,
            weights_only=False,
            config=config
        )
    elif config.checkpoint.pretrain_weights is not None:
        # Loads only encoder weights for segmentation
        print("Loading pretrained encoder weights...")

        checkpoint = torch.load(config.checkpoint.pretrain_weights, weights_only=False, map_location="cpu")
        state_dict = checkpoint["state_dict"]

        for key in list(state_dict.keys()):
            new_key = key.replace("backbone.", "")
            state_dict[new_key] = state_dict.pop(key)
        
        model.backbone.load_state_dict(state_dict, strict=True)
    
    return model


def get_random_embedding(patch_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Return a random embedding of the given patch embeddings that is not the cls token.
    """
    if len(patch_embeddings.shape) == 4:
        patch_embeddings = rearrange(patch_embeddings, "b h w d -> b (h w) d")

    assert len(patch_embeddings.shape) == 3, "Expected patch_embeddings to be of shape (batch_size, num_patches, embedding_dim)"
    
    num_patches = patch_embeddings.shape[1]
    patch_index = torch.randint(1, num_patches, (1,)).item()
    
    return patch_embeddings[:, patch_index]


def reconstruct_full_image(
        outputs: list[dict[str, torch.Tensor]],
        full_shape: tuple[int, int],
        num_classes: int,
        patch_size: int = 96
        ) -> torch.Tensor:
    H, W = full_shape

    full_logits = torch.zeros((num_classes, H, W), dtype=torch.float32)
    count_map = torch.zeros((H, W), dtype=torch.float32)

    for batch_out in outputs:
        logits = batch_out["logits"]   # (B, C, H, W)
        coords = batch_out["coords"]   # (B, 2)

        for i in range(logits.size(0)):
            _, top, left = coords[i].tolist()

            full_logits[:, top:top+patch_size, left:left+patch_size] += logits[i].cpu()
            count_map[top:top+patch_size, left:left+patch_size] += 1
    
    # Normalizing accumulated logits
    full_logits /= count_map.clamp_min(1e-6)

    # Softmax
    probs = full_logits.softmax(dim=0)
    pred = probs.argmax(dim=0)

    return pred


def superpixel_majority_vote(unet_preds: torch.Tensor, superpixel_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Refines a batch of U-Net predictions using superpixel boundaries via majority voting.

    Args:
        unet_preds (torch.Tensor): Batched discrete class predictions of shape (B, H, W).
                                   Expected to be integers (e.g., torch.long).
        superpixel_labels (torch.Tensor): Batched superpixel IDs of shape (B, H, W).
        num_classes (int): Total number of distinct classes in your dataset.

    Returns:
        torch.Tensor: Refined predictions of shape (B, H, W), matching the input device.
    """
    unet_preds = unet_preds.long()
    superpixel_labels = superpixel_labels.long()

    B, H, W = unet_preds.shape
    
    refined_preds = torch.zeros_like(unet_preds)

    for b in range(B):
        preds_b = unet_preds[b].flatten()
        sp_b = superpixel_labels[b].flatten()
        
        unique_sp, inverse_indices = torch.unique(sp_b, return_inverse=True)
        num_sp = unique_sp.size(0)
        
        linear_indices = inverse_indices * num_classes + preds_b
        
        counts = torch.bincount(linear_indices, minlength=num_sp * num_classes)
        counts = counts.view(num_sp, num_classes)
        
        majority_classes = torch.argmax(counts, dim=1)
        
        refined_b = majority_classes[inverse_indices].view(H, W)
        refined_preds[b] = refined_b
        
    return refined_preds.unsqueeze(1)