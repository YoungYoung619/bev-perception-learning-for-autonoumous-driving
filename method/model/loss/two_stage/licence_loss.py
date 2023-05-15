from torch import nn
from .cls_loss import LabelSmoothingLoss
import torch

# class LicenceLoss(nn.Module):
#     def __init__(self):
#         super(LicenceLoss, self).__init__()
#         self.cls_loss = LabelSmoothingLoss(smoothing=0.01)
#
#     def generate_gt(self, preds, gt_meta, input_size):
#         exist_mask = gt_meta[0]
#         kps = torch.stack(gt_meta[1], dim=-1)
#         bs, n_attr, nh, nw = preds.size()
#         kps = kps.reshape(bs, -1, 2)
#         iw, ih = input_size
#
#         device = preds.device
#         gt_licence = torch.zeros(size=(bs, 9, nh, nw), dtype=torch.float32, device=device)
#         # rescale to h w
#         x_gt = kps[..., 0] / iw * nw
#         y_gt = kps[..., 1] / ih * nh
#
#         xc = x_gt.mean(dim=-1)
#         yc = y_gt.mean(dim=-1)
#         x_idx = xc.long().clip(0, nw - 1)
#         y_idx = yc.long().clip(0, nh - 1)
#         x_idx_c = x_idx.float() + 0.5
#         y_idx_c = y_idx.float() + 0.5
#         batch_idx = torch.arange(0, bs).to(device)
#
#         x_offset = x_gt - x_idx_c[:, None]
#         y_offset = y_gt - y_idx_c[:, None]
#
#         gt_licence[batch_idx[exist_mask], 0, y_idx[exist_mask], x_idx[exist_mask]] = 1
#         gt_licence[batch_idx[exist_mask], 1, y_idx[exist_mask], x_idx[exist_mask]] = x_offset[:, 0][exist_mask]
#         gt_licence[batch_idx[exist_mask], 2, y_idx[exist_mask], x_idx[exist_mask]] = y_offset[:, 0][exist_mask]
#         gt_licence[batch_idx[exist_mask], 3, y_idx[exist_mask], x_idx[exist_mask]] = x_offset[:, 1][exist_mask]
#         gt_licence[batch_idx[exist_mask], 4, y_idx[exist_mask], x_idx[exist_mask]] = y_offset[:, 1][exist_mask]
#         gt_licence[batch_idx[exist_mask], 5, y_idx[exist_mask], x_idx[exist_mask]] = x_offset[:, 2][exist_mask]
#         gt_licence[batch_idx[exist_mask], 6, y_idx[exist_mask], x_idx[exist_mask]] = y_offset[:, 2][exist_mask]
#         gt_licence[batch_idx[exist_mask], 7, y_idx[exist_mask], x_idx[exist_mask]] = x_offset[:, 3][exist_mask]
#         gt_licence[batch_idx[exist_mask], 8, y_idx[exist_mask], x_idx[exist_mask]] = y_offset[:, 3][exist_mask]
#
#         return gt_licence
#
#     def forward(self, preds, gt_meta, input_size):
#         """ loss calculate
#         Args:
#         """
#         gt_licence = self.generate_gt(preds, gt_meta, input_size)
#
#         clf_gt = gt_licence[:, 0, :, :].reshape(-1).long()
#         clf_pred = preds[:, :2, :, :].permute(0, 2, 3, 1).reshape(-1, 2)
#         clf_loss = self.cls_loss(clf_pred, clf_gt)
#
#         pos_mask = gt_licence[:, 0, :, :] == 1
#         coord_gt = gt_licence[:, 1:, :, :].permute(0, 2, 3, 1)
#         coord_gt = coord_gt[pos_mask]
#
#         coord_pred = preds[:, 2:, :, :].permute(0, 2, 3, 1)
#         coord_pred = coord_pred[pos_mask]
#
#         coord_loss = torch.abs(coord_pred - coord_gt).mean()
#
#         licence_loss = clf_loss + coord_loss
#         return licence_loss, {"licence_clf_loss": clf_loss,
#                              "licence_coord_loss": coord_loss,
#                              "licence_loss": licence_loss}


class LicenceLoss(nn.Module):
    def __init__(self):
        super(LicenceLoss, self).__init__()
        self.cls_loss = LabelSmoothingLoss(smoothing=0.01)

    def generate_gt(self, preds, gt_meta, input_size):
        iw, ih = input_size
        exist_mask = torch.cat(gt_meta[0], dim=0)
        boxes_licence = torch.stack(gt_meta[1][0], dim=-1)
        boxes_attach_licence = torch.stack(gt_meta[1][1], dim=-1)
        licence_total = torch.cat([boxes_licence, boxes_attach_licence], dim=0)[exist_mask]

        bs, n_attr, nh, nw = preds.size()
        x1_gt = licence_total[:, 0] / iw * nw
        y1_gt = licence_total[:, 1] / ih * nh
        x2_gt = licence_total[:, 2] / iw * nw
        y2_gt = licence_total[:, 3] / ih * nh

        xc = (x1_gt + x2_gt) / 2
        yc = (y1_gt + y2_gt) / 2
        x_idx = xc.long().clip(0, nw - 1)
        y_idx = yc.long().clip(0, nh - 1)

        device = preds.device
        gt_licence = torch.zeros(size=(bs, 5, nh, nw), dtype=torch.float32, device=device)

        batch_idx = torch.arange(0, bs).to(device).repeat(2)[exist_mask]
        gt_licence[batch_idx, 0, y_idx, x_idx] = 1
        gt_licence[batch_idx, 1, y_idx, x_idx] = xc - x_idx.float()
        gt_licence[batch_idx, 2, y_idx, x_idx] = yc - y_idx.float()
        gt_licence[batch_idx, 3, y_idx, x_idx] = (x2_gt - x1_gt) / nw       # 归一化尺寸
        gt_licence[batch_idx, 4, y_idx, x_idx] = (y2_gt - y1_gt) / nh       # 归一化尺寸

        return gt_licence

    def forward(self, preds, gt_meta, input_size):
        """ loss calculate
        Args:
        """
        gt_licence = self.generate_gt(preds, gt_meta, input_size)

        clf_gt = gt_licence[:, 0, :, :].reshape(-1).long()
        clf_pred = preds[:, :2, :, :].permute(0, 2, 3, 1).reshape(-1, 2)
        clf_loss = self.cls_loss(clf_pred, clf_gt)

        pos_mask = gt_licence[:, 0, :, :] == 1
        coord_gt = gt_licence[:, 1:, :, :].permute(0, 2, 3, 1)
        coord_gt = coord_gt[pos_mask]

        coord_pred = preds[:, 2:, :, :].permute(0, 2, 3, 1)
        coord_pred = coord_pred[pos_mask]

        coord_loss = torch.abs(coord_pred - coord_gt).mean()

        licence_loss = clf_loss + coord_loss
        return licence_loss, {"licence_clf_loss": clf_loss,
                             "licence_coord_loss": coord_loss,
                             "licence_loss": licence_loss}
