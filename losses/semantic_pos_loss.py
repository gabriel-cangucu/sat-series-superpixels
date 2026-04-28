import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticPosLoss(nn.Module):
    def __init__(self, pos_weight: float = 0.03, kernel_size: int = 16) -> None:
        super().__init__()

        self.pos_weight = pos_weight
        self.kernel_size = kernel_size

    def forward(self, probs: torch.Tensor, targets: torch.Tensor, XY_features: torch.Tensor) -> torch.Tensor:
        # This wrt the slic paper who used sqrt of (mse)

        # rgbxy1_feat: B*50+2*H*W
        # output : B*9*H*w
        # NOTE: this loss is only designed for one level structure

        # Todo: currently we assume the downsize scale in x,y direction are always same
        S = self.kernel_size
        m = self.pos_weight
        b = len(probs)
        prob = probs.clone()

        XY_features = torch.tile(XY_features, (b, 1, 1, 1))
        labxy_feat = self._build_LABXY_feat(targets, XY_features)

        b, c, h, w = labxy_feat.shape
        pooled_labxy = self._poolfeat(labxy_feat, prob, self.kernel_size, self.kernel_size)
        reconstr_feat = self._upfeat(pooled_labxy, prob, self.kernel_size, self.kernel_size)

        loss_map = reconstr_feat[:,-2:,:,:] - labxy_feat[:,-2:,:,:]

        # Self def cross entropy  -- the official one combined softmax
        logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-8)
        loss_sem = - torch.sum(logit * labxy_feat[:, :-2, :, :]) / b
        loss_pos = torch.norm(loss_map, p=2, dim=1).sum() / b * m

        # Empirically we find timing 0.005 tend to better performance
        loss_sum =  0.005 * (loss_sem + loss_pos)
        # loss_sem_sum =  0.005 * loss_sem
        # loss_pos_sum = 0.005 * loss_pos

        return loss_sum

    def _build_LABXY_feat(self, targets: torch.Tensor, XY_features: torch.Tensor) -> torch.Tensor:
        targets_one_hot = self._label_to_one_hot(targets, C=50)
        img_lab = targets_one_hot.clone().type(torch.float)

        b, _, curr_img_height, curr_img_width = XY_features.shape
        scale_img =  F.interpolate(img_lab, size=(curr_img_height, curr_img_width), mode='nearest')
        LABXY_feat = torch.cat([scale_img, XY_features], dim=1)

        return LABXY_feat

    def _label_to_one_hot(self, labels: torch.Tensor, C: int = 14) -> torch.Tensor:
        # w.r.t http://jacobkimmel.github.io/pytorch_onehot/
        '''
        Converts an integer label torch.autograd.Variable to a one-hot Variable.

        Parameters
        ----------
        labels : torch.autograd.Variable of torch.cuda.LongTensor
            N x 1 x H x W, where N is batch size.
            Each value is an integer representing correct classification.
        C : integer.
            number of classes in labels.

        Returns
        -------
        target : torch.cuda.FloatTensor
            N x C x H x W, where C is class number. One-hot encoded.
        '''
        b, _, h, w = labels.shape
        one_hot = torch.zeros(b, C, h, w, dtype=torch.long).cuda()
        target = one_hot.scatter_(1, labels.type(torch.long).data, 1) #require long type

        return target.type(torch.float32)

    def _poolfeat(self, input: torch.Tensor, prob: torch.Tensor, sp_h: int = 2, sp_w: int = 2) -> torch.Tensor:

        def feat_prob_sum(feat_sum, prob_sum, shift_feat):
            feat_sum += shift_feat[:, :-1, :, :]
            prob_sum += shift_feat[:, -1:, :, :]
            return feat_sum, prob_sum

        b, _, h, w = input.shape

        h_shift_unit = 1
        w_shift_unit = 1
        p2d = (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit)
        feat_ = torch.cat([input, torch.ones([b, 1, h, w]).cuda()], dim=1)  # b* (n+1) *h*w
        prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 0, 1), kernel_size=(sp_h, sp_w),stride=(sp_h, sp_w)) # b * (n+1) * h* w
        send_to_top_left =  F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, 2 * w_shift_unit:]
        feat_sum = send_to_top_left[:, :-1, :, :].clone()
        prob_sum = send_to_top_left[:, -1:, :, :].clone()

        prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 1, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
        top = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, w_shift_unit:-w_shift_unit]
        feat_sum, prob_sum = feat_prob_sum(feat_sum,prob_sum,top )

        prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 2, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
        top_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, :-2 * w_shift_unit]
        feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top_right)

        prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 3, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
        left = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, h_shift_unit:-h_shift_unit, 2 * w_shift_unit:]
        feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, left)

        prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 4, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
        center = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, w_shift_unit:-w_shift_unit]
        feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, center)

        prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 5, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
        right = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, h_shift_unit:-h_shift_unit, :-2 * w_shift_unit]
        feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, right)

        prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 6, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
        bottom_left = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, :-2 * h_shift_unit, 2 * w_shift_unit:]
        feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_left)

        prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 7, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
        bottom = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, w_shift_unit:-w_shift_unit]
        feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom)

        prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 8, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
        bottom_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, :-2 * w_shift_unit]
        feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_right)

        pooled_feat = feat_sum / (prob_sum + 1e-8)

        return pooled_feat
    
    def _upfeat(self, input: torch.Tensor, prob: torch.Tensor, up_h: int = 2, up_w: int = 2) -> torch.Tensor:
        # input b*n*H*W  downsampled
        # prob b*9*h*w
        b, c, h, w = input.shape

        h_shift = 1
        w_shift = 1

        p2d = (w_shift, w_shift, h_shift, h_shift)
        feat_pd = F.pad(input, p2d, mode='constant', value=0)

        gt_frm_top_left = F.interpolate(feat_pd[:, :, :-2 * h_shift, :-2 * w_shift], size=(h * up_h, w * up_w),mode='nearest')
        feat_sum = gt_frm_top_left * prob.narrow(1,0,1)

        top = F.interpolate(feat_pd[:, :, :-2 * h_shift, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
        feat_sum += top * prob.narrow(1, 1, 1)

        top_right = F.interpolate(feat_pd[:, :, :-2 * h_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
        feat_sum += top_right * prob.narrow(1,2,1)

        left = F.interpolate(feat_pd[:, :, h_shift:-w_shift, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
        feat_sum += left * prob.narrow(1, 3, 1)

        center = F.interpolate(input, (h * up_h, w * up_w), mode='nearest')
        feat_sum += center * prob.narrow(1, 4, 1)

        right = F.interpolate(feat_pd[:, :, h_shift:-w_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
        feat_sum += right * prob.narrow(1, 5, 1)

        bottom_left = F.interpolate(feat_pd[:, :, 2 * h_shift:, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
        feat_sum += bottom_left * prob.narrow(1, 6, 1)

        bottom = F.interpolate(feat_pd[:, :, 2 * h_shift:, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
        feat_sum += bottom * prob.narrow(1, 7, 1)

        bottom_right =  F.interpolate(feat_pd[:, :, 2 * h_shift:, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
        feat_sum += bottom_right * prob.narrow(1, 8, 1)

        return feat_sum