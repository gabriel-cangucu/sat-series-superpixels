import torch
import torch.nn.functional as F
import numpy as np
from skimage.segmentation._slic import _enforce_label_connectivity_cython
# from .cython.connectivity import enforce_connectivity


def init_superpixel_grid(img_height: int, img_width: int, kernel_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    def _shift9pos(input: np.ndarray, h_shift_unit: int = 1,  w_shift_unit: int = 1):
        # Input should be padding as (c, 1+ height+1, 1+width+1)
        input_pd = np.pad(input, ((h_shift_unit, h_shift_unit), (w_shift_unit, w_shift_unit)), mode='edge')
        input_pd = np.expand_dims(input_pd, axis=0)

        # Assign to ...
        top     = input_pd[:, :-2 * h_shift_unit,          w_shift_unit:-w_shift_unit]
        bottom  = input_pd[:, 2 * h_shift_unit:,           w_shift_unit:-w_shift_unit]
        left    = input_pd[:, h_shift_unit:-h_shift_unit,  :-2 * w_shift_unit]
        right   = input_pd[:, h_shift_unit:-h_shift_unit,  2 * w_shift_unit:]

        center = input_pd[:,h_shift_unit:-h_shift_unit,w_shift_unit:-w_shift_unit]

        bottom_right    = input_pd[:, 2 * h_shift_unit:,   2 * w_shift_unit:]
        bottom_left     = input_pd[:, 2 * h_shift_unit:,   :-2 * w_shift_unit]
        top_right       = input_pd[:, :-2 * h_shift_unit,  2 * w_shift_unit:]
        top_left        = input_pd[:, :-2 * h_shift_unit,  :-2 * w_shift_unit]

        shift_tensor = np.concatenate([top_left,    top,      top_right,
                                       left,        center,      right,
                                       bottom_left, bottom,    bottom_right], axis=0)
        
        return shift_tensor
    
    # Get spixel id for the final assignment
    n_spixl_h = int(np.floor(img_height / kernel_size))
    n_spixl_w = int(np.floor(img_width / kernel_size))

    spixel_height = int(img_height / (1. * n_spixl_h))
    spixel_width = int(img_width / (1. * n_spixl_w))

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = _shift9pos(spix_values)

    spix_idx_tensor =  np.repeat(
        np.repeat(spix_idx_tensor_, spixel_height, axis=1), spixel_width, axis=2
    )
    superpixel_map = torch.from_numpy(spix_idx_tensor.astype(np.float32)).cuda()

    curr_img_height = int(np.floor(img_height))
    curr_img_width = int(np.floor(img_width))

    # Pixel coord
    all_h_coords = np.arange(0, curr_img_height, 1) / curr_img_height
    all_w_coords = np.arange(0, curr_img_width, 1) / curr_img_width
    curr_pxl_coord = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing='ij'))

    coord_tensor = np.concatenate([curr_pxl_coord[1:2, :, :], curr_pxl_coord[:1, :, :]])
    XY_features = torch.from_numpy(coord_tensor.astype(np.float32)).cuda()

    return superpixel_map, XY_features


def update_superpixel_map(superpixel_map_in: torch.Tensor, assign_map_in: torch.Tensor) -> torch.Tensor:
    assign_map = assign_map_in.clone()
    b, _, h, w = assign_map.shape

    superpixel_map_in = torch.tile(superpixel_map_in, (b, 1, 1, 1))
    _, _, id_h, id_w = superpixel_map_in.shape

    if (id_h == h) and (id_w == w):
        spixl_map_idx = superpixel_map_in
    else:
        spixl_map_idx = F.interpolate(superpixel_map_in, size=(h, w), mode='nearest')

    assig_max, _ = torch.max(assign_map, dim=1, keepdim=True)
    assignment_ = torch.where(assign_map == assig_max, torch.ones(assign_map.shape).cuda(),torch.zeros(assign_map.shape).cuda())
    new_spixl_map_ = spixl_map_idx * assignment_ # winner takes all
    new_spixl_map = torch.sum(new_spixl_map_, dim=1, keepdim=True).type(torch.int)

    return new_spixl_map


def post_process_superpixels(superpixel_maps: torch.Tensor, num_superpixels: int = 169) -> torch.Tensor:
    for i in range(len(superpixel_maps)):
        index_map_numpy = superpixel_maps[i].squeeze().detach().cpu().numpy()
        index_map_numpy = index_map_numpy.astype(np.int64)

        segment_size = (index_map_numpy.shape[0] * index_map_numpy.shape[1]) / (int(num_superpixels) * 1.0)
        min_size = int(0.06 * segment_size)
        max_size =  int(3 * segment_size)

        index_map_numpy = _enforce_label_connectivity_cython(index_map_numpy[None, :, :], min_size, max_size)[0]
        superpixel_maps[i] = torch.tensor(index_map_numpy)

    return superpixel_maps