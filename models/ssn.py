import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

from .models_utils.torch_utils import conv_bn_relu
from .models_utils.pair_wise_distance_cuda_source import source
from losses.reconstruction_loss import ReconstructionLoss
from utils.superpixels_utils import post_process_superpixels


pair_wise_distance_cuda = load_inline(
    "pair_wise_distance", cpp_sources="", cuda_sources=source
)


class SSN(nn.Module):
    def __init__(self,
            img_size,
            num_channels,
            num_frames,
            feature_dim = 20,
            nspix = 100,
            pos_scale = 10,
            n_iter = 5
        ):
        super().__init__()
        self.img_size = img_size
        self.nspix = nspix
        self.pos_scale = pos_scale
        self.n_iter = n_iter

        num_features = (num_channels * num_frames) + 2

        self.scale1 = nn.Sequential(
            conv_bn_relu(num_features, 64),
            conv_bn_relu(64, 64)
        )
        self.scale2 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64)
        )
        self.scale3 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64)
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(64*3+num_features, feature_dim-5, 3, padding=1),
            nn.ReLU(True)
        )

        self.coords = torch.stack(
            torch.meshgrid(torch.arange(img_size), torch.arange(img_size), indexing="ij"), dim=0
        )

        self.criterion = ReconstructionLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        coords = self.coords[None].repeat(x.shape[0], 1, 1, 1).float()
        coords = (coords / self.img_size).to(x.device)

        x = torch.cat([x, self.pos_scale * coords], dim=1)
        pixel_f = self.feature_extract(x)

        if self.training:
            abs_affinity, hard_labels, _ =  self._ssn_iter(pixel_f, self.nspix, self.n_iter)
        else:
            abs_affinity, hard_labels, _ = self._sparse_ssn_iter(pixel_f, self.nspix, self.n_iter)
        
        labels = hard_labels.reshape(hard_labels.shape[0], self.img_size, self.img_size)
        labels = post_process_superpixels(labels).unsqueeze(1)

        if abs_affinity.is_sparse:
            abs_affinity = abs_affinity.to_dense()

        loss = self.criterion(abs_affinity, y, coords, hard_labels)

        return {
            "preds": labels,
            "loss": loss
        }

    def feature_extract(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(s1)
        s3 = self.scale3(s2)

        s2 = nn.functional.interpolate(s2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        s3 = nn.functional.interpolate(s3, size=s1.shape[-2:], mode="bilinear", align_corners=False)

        cat_feat = torch.cat([x, s1, s2, s3], 1)
        feat = self.output_conv(cat_feat)

        return torch.cat([feat, x], 1)
    
    def _ssn_iter(self, pixel_features, num_spixels, n_iter):
        """
        Computing assignment iterations
        detailed process is in Algorithm 1, line 2 - 6

        Args:
            pixel_features: torch.Tensor
                A Tensor of shape (B, C, H, W)
            num_spixels: int
                A number of superpixels
            n_iter: int
                A number of iterations
            return_hard_label: bool
                return hard assignment or not
        """
        height, width = pixel_features.shape[-2:]
        num_spixels_width = int(math.sqrt(num_spixels * width / height))
        num_spixels_height = int(math.sqrt(num_spixels * height / width))

        spixel_features, init_label_map = self._calc_init_centroid(pixel_features,
                                                                   num_spixels_width,
                                                                   num_spixels_height)
        abs_indices = self._get_abs_indices(init_label_map, num_spixels_width)

        pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1)
        permuted_pixel_features = pixel_features.permute(0, 2, 1).contiguous()

        for _ in range(n_iter):
            dist_matrix = PairwiseDistFunction.apply(
                pixel_features, spixel_features, init_label_map, num_spixels_width, num_spixels_height)

            affinity_matrix = (-dist_matrix).softmax(1)
            reshaped_affinity_matrix = affinity_matrix.reshape(-1)

            actual_num_spixels = num_spixels_width * num_spixels_height
            mask = (abs_indices[1] >= 0) & (abs_indices[1] < actual_num_spixels)

            sparse_abs_affinity = torch.sparse_coo_tensor(
                abs_indices[:, mask],
                reshaped_affinity_matrix[mask],
                size=(pixel_features.shape[0], actual_num_spixels, pixel_features.shape[-1])
            )

            abs_affinity = sparse_abs_affinity.to_dense().contiguous()
            spixel_features = torch.bmm(abs_affinity, permuted_pixel_features) \
                / (abs_affinity.sum(2, keepdim=True) + 1e-16)

            spixel_features = spixel_features.permute(0, 2, 1).contiguous()

        hard_labels = self._get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width)

        return abs_affinity, hard_labels, spixel_features
    
    @torch.no_grad()
    def _sparse_ssn_iter(self, pixel_features, num_spixels, n_iter):
        """
        computing assignment iterations with sparse matrix
        detailed process is in Algorithm 1, line 2 - 6
        NOTE: this function does NOT guarantee the backward computation.

        Args:
            pixel_features: torch.Tensor
                A Tensor of shape (B, C, H, W)
            num_spixels: int
                A number of superpixels
            n_iter: int
                A number of iterations
            return_hard_label: bool
                return hard assignment or not
        """
        height, width = pixel_features.shape[-2:]
        num_spixels_width = int(math.sqrt(num_spixels * width / height))
        num_spixels_height = int(math.sqrt(num_spixels * height / width))

        spixel_features, init_label_map = self._calc_init_centroid(pixel_features,
                                                                  num_spixels_width,
                                                                  num_spixels_height)
        abs_indices = self._get_abs_indices(init_label_map, num_spixels_width)

        pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1)
        permuted_pixel_features = pixel_features.permute(0, 2, 1)

        for _ in range(n_iter):
            dist_matrix = PairwiseDistFunction.apply(
                pixel_features, spixel_features, init_label_map, num_spixels_width, num_spixels_height)

            affinity_matrix = (-dist_matrix).softmax(1)
            reshaped_affinity_matrix = affinity_matrix.reshape(-1)

            actual_num_spixels = num_spixels_width * num_spixels_height
            mask = (abs_indices[1] >= 0) & (abs_indices[1] < actual_num_spixels)

            sparse_abs_affinity = torch.sparse_coo_tensor(
                abs_indices[:, mask],
                reshaped_affinity_matrix[mask],
                size=(pixel_features.shape[0], actual_num_spixels, pixel_features.shape[-1])
            )

            spixel_features = self._naive_sparse_bmm(sparse_abs_affinity, permuted_pixel_features) \
                / (torch.sparse.sum(sparse_abs_affinity, 2).to_dense()[..., None] + 1e-16)

            spixel_features = spixel_features.permute(0, 2, 1)

        hard_labels = self._get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width)

        return sparse_abs_affinity, hard_labels, spixel_features

    def _naive_sparse_bmm(self, sparse_mat, dense_mat, transpose=False):
        if transpose:
            return torch.stack([torch.sparse.mm(s_mat, d_mat.t()) for s_mat, d_mat in zip(sparse_mat, dense_mat)], 0)
        else:
            return torch.stack([torch.sparse.mm(s_mat, d_mat) for s_mat, d_mat in zip(sparse_mat, dense_mat)], 0)
    
    def _calc_init_centroid(self, images, num_spixels_width, num_spixels_height):
        """
        calculate initial superpixels

        Args:
            images: torch.Tensor
                A Tensor of shape (B, C, H, W)
            spixels_width: int
                initial superpixel width
            spixels_height: int
                initial superpixel height

        Return:
            centroids: torch.Tensor
                A Tensor of shape (B, C, H * W)
            init_label_map: torch.Tensor
                A Tensor of shape (B, H * W)
            num_spixels_width: int
                A number of superpixels in each column
            num_spixels_height: int
                A number of superpixels int each raw
        """
        batchsize, channels, height, width = images.shape
        device = images.device

        centroids = torch.nn.functional.adaptive_avg_pool2d(images, (num_spixels_height, num_spixels_width))

        with torch.no_grad():
            num_spixels = num_spixels_width * num_spixels_height
            labels = torch.arange(num_spixels, device=device).reshape(1, 1, *centroids.shape[-2:]).type_as(centroids)
            init_label_map = torch.nn.functional.interpolate(labels, size=(height, width), mode="nearest")
            init_label_map = init_label_map.repeat(batchsize, 1, 1, 1)

        init_label_map = init_label_map.reshape(batchsize, -1)
        centroids = centroids.reshape(batchsize, channels, -1)

        return centroids, init_label_map

    @torch.no_grad()
    def _get_abs_indices(self, init_label_map, num_spixels_width):
        b, n_pixel = init_label_map.shape
        device = init_label_map.device
        r = torch.arange(-1, 2.0, device=device)
        relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)

        abs_pix_indices = torch.arange(n_pixel, device=device)[None, None].repeat(b, 9, 1).reshape(-1).long()
        abs_spix_indices = (init_label_map[:, None] + relative_spix_indices[None, :, None]).reshape(-1).long()
        abs_batch_indices = torch.arange(b, device=device)[:, None, None].repeat(1, 9, n_pixel).reshape(-1).long()

        return torch.stack([abs_batch_indices, abs_spix_indices, abs_pix_indices], 0)
    
    @torch.no_grad()
    def _get_hard_abs_labels(self, affinity_matrix, init_label_map, num_spixels_width):
        relative_label = affinity_matrix.max(1)[1]
        r = torch.arange(-1, 2.0, device=affinity_matrix.device)
        relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)
        label = init_label_map + relative_spix_indices[relative_label]

        max_spixel_idx = init_label_map.max()
        label = label.clamp(0, max_spixel_idx)

        return label.long()


class PairwiseDistFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pixel_features, spixel_features, init_spixel_indices, num_spixels_width, num_spixels_height):
        ctx.num_spixels_width = num_spixels_width
        ctx.num_spixels_height = num_spixels_height
        output = pixel_features.new(pixel_features.shape[0], 9, pixel_features.shape[-1]).zero_()
        ctx.save_for_backward(pixel_features, spixel_features, init_spixel_indices)

        return pair_wise_distance_cuda.forward(
            pixel_features.contiguous(), spixel_features.contiguous(),
            init_spixel_indices.contiguous(), output,
            ctx.num_spixels_width, ctx.num_spixels_height)

    @staticmethod
    def backward(ctx, dist_matrix_grad):
        pixel_features, spixel_features, init_spixel_indices = ctx.saved_tensors

        pixel_features_grad = torch.zeros_like(pixel_features)
        spixel_features_grad = torch.zeros_like(spixel_features)
        
        pixel_features_grad, spixel_features_grad = pair_wise_distance_cuda.backward(
            dist_matrix_grad.contiguous(), pixel_features.contiguous(),
            spixel_features.contiguous(), init_spixel_indices.contiguous(),
            pixel_features_grad, spixel_features_grad,
            ctx.num_spixels_width, ctx.num_spixels_height
        )
        return pixel_features_grad, spixel_features_grad, None, None, None


if __name__ == "__main__":
    model = SSN(num_channels=4, num_frames=1, img_size=208).cuda()

    x = torch.randn(16, 4*1, 208, 208).cuda()
    y = torch.zeros(16, 1, 208, 208).cuda()

    outputs = model(x, y)

    print(outputs["preds"].shape)
    print(outputs["loss"])