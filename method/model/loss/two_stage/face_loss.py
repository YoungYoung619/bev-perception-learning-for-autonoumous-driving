from torch import nn
from .cls_loss import LabelSmoothingLoss
import torch


class FaceLoss(nn.Module):
    """
    物体中心落在哪，就由那个网格去预测此物体的中心偏移，及高宽
    """
    def __init__(self):
        super(FaceLoss, self).__init__()
        self.cls_loss = LabelSmoothingLoss(smoothing=0.01)

    def generate_gt(self, preds, gt_meta, input_size):
        iw, ih = input_size
        exist_mask = torch.stack(gt_meta[0], dim=1)
        # boxes_face = torch.stack(gt_meta[1], dim=0)
        boxes = []
        for box_each in gt_meta[1]:
            box_each = torch.stack(box_each, dim=-1)
            boxes.append(box_each)
        face_boxes = torch.stack(boxes, dim=1)[exist_mask]

        bs, n_attr, nh, nw = preds.size()
        x1_gt = face_boxes[:, 0] / iw * nw
        y1_gt = face_boxes[:, 1] / ih * nh
        x2_gt = face_boxes[:, 2] / iw * nw
        y2_gt = face_boxes[:, 3] / ih * nh

        xc = (x1_gt + x2_gt) / 2
        yc = (y1_gt + y2_gt) / 2
        x_idx = xc.long().clip(0, nw - 1)
        y_idx = yc.long().clip(0, nh - 1)

        device = preds.device
        gt_face = torch.zeros(size=(bs, 5, nh, nw), dtype=torch.float32, device=device)
        batch_idx = torch.arange(0, bs)[:, None].repeat(1, 10)[exist_mask]
        gt_face[batch_idx, 0, y_idx, x_idx] = 1
        gt_face[batch_idx, 1, y_idx, x_idx] = xc - x_idx.float()
        gt_face[batch_idx, 2, y_idx, x_idx] = yc - y_idx.float()
        gt_face[batch_idx, 3, y_idx, x_idx] = (x2_gt - x1_gt) / nw  # 归一化尺寸
        gt_face[batch_idx, 4, y_idx, x_idx] = (y2_gt - y1_gt) / nh  # 归一化尺寸

        return gt_face

    def forward(self, preds, gt_meta, input_size):
        """ loss calculate
        Args:
        """
        gt_face = self.generate_gt(preds, gt_meta, input_size)

        clf_gt = gt_face[:, 0, :, :].reshape(-1).long()
        clf_pred = preds[:, :2, :, :].permute(0, 2, 3, 1).reshape(-1, 2)
        clf_loss = self.cls_loss(clf_pred, clf_gt)

        pos_mask = gt_face[:, 0, :, :] == 1
        a = pos_mask.sum()
        coord_gt = gt_face[:, 1:, :, :].permute(0, 2, 3, 1)
        coord_gt = coord_gt[pos_mask]

        coord_pred = preds[:, 2:, :, :].permute(0, 2, 3, 1)
        coord_pred = coord_pred[pos_mask]

        coord_loss = torch.abs(coord_pred - coord_gt).mean()

        face_loss = clf_loss + coord_loss
        return face_loss, {"face_clf_loss": clf_loss,
                           "face_coord_loss": coord_loss,
                           "face_loss": face_loss}


class FaceMultiMatchLoss(FaceLoss):
    """
    物体中心和周围的一些网格，都会用来预测此物体的中心偏移，及高宽
    """
    def __init__(self):
        super(FaceMultiMatchLoss, self).__init__()
        self.window_size = 3
        self.dis = self.window_size // 2

    def generate_gt(self, preds, gt_meta, input_size):
        iw, ih = input_size
        exist_mask = torch.stack(gt_meta[0], dim=1)
        # boxes_face = torch.stack(gt_meta[1], dim=0)
        boxes = []
        for box_each in gt_meta[1]:
            box_each = torch.stack(box_each, dim=-1)
            boxes.append(box_each)
        face_boxes = torch.stack(boxes, dim=1)[exist_mask]

        device = preds.device
        bs, n_attr, nh, nw = preds.size()
        x1_gt = face_boxes[:, 0] / iw * nw
        y1_gt = face_boxes[:, 1] / ih * nh
        x2_gt = face_boxes[:, 2] / iw * nw
        y2_gt = face_boxes[:, 3] / ih * nh

        xc = (x1_gt + x2_gt) / 2
        yc = (y1_gt + y2_gt) / 2
        x_idx = xc.long().clip(0, nw - 1)
        y_idx = yc.long().clip(0, nh - 1)

        grid_x, grid_y = torch.meshgrid(torch.arange(-self.dis, nw + self.dis).to(device),
                                        torch.arange(-self.dis, nh + self.dis).to(device))
        # gr2id_x, grid_y = torch.meshgrid(torch.arange(0, nw), torch.arange(0, nh))
        grid_x = grid_x.transpose(0, 1)
        grid_y = grid_y.transpose(0, 1)
        grid_xy = torch.stack([grid_x, grid_y], dim=-1)
        grid_x = grid_xy[None, :, :, 0].repeat(bs, 1, 1, 1)
        grid_y = grid_xy[None, :, :, 1].repeat(bs, 1, 1, 1)
        grid_x = grid_x.float()
        grid_y = grid_y.float()

        grid_x_win_patchs = torch.nn.functional.unfold(grid_x, kernel_size=self.window_size,
                                                       stride=1, padding=0)
        grid_y_win_patchs = torch.nn.functional.unfold(grid_y, kernel_size=self.window_size,
                                                       stride=1, padding=0)

        grid_x_win_patchs = grid_x_win_patchs.reshape(bs, -1, nh, nw).clip(0, nw - 1)
        grid_y_win_patchs = grid_y_win_patchs.reshape(bs, -1, nh, nw).clip(0, nh - 1)

        batch_idxs = torch.arange(0, bs)[:, None].repeat(1, 10)[exist_mask]

        grid_x_idxs = grid_x_win_patchs[batch_idxs, :, y_idx, x_idx].long()
        grid_y_idxs = grid_y_win_patchs[batch_idxs, :, y_idx, x_idx].long()
        batch_idxs = batch_idxs[:, None].repeat(1, self.window_size * self.window_size)

        xcs = xc[:, None].repeat(1, self.window_size * self.window_size)
        ycs = yc[:, None].repeat(1, self.window_size * self.window_size)
        x1_gts = x1_gt[:, None].repeat(1, self.window_size * self.window_size)
        y1_gts = y1_gt[:, None].repeat(1, self.window_size * self.window_size)
        x2_gts = x2_gt[:, None].repeat(1, self.window_size * self.window_size)
        y2_gts = y2_gt[:, None].repeat(1, self.window_size * self.window_size)
        gt_face = torch.zeros(size=(bs, 5, nh, nw), dtype=torch.float32, device=device)
        gt_face[batch_idxs, 0, grid_y_idxs, grid_x_idxs] = 1
        gt_face[batch_idxs, 1, grid_y_idxs, grid_x_idxs] = (xcs - (grid_x_idxs.float() + 0.5)) / (self.window_size / 2.)
        gt_face[batch_idxs, 2, grid_y_idxs, grid_x_idxs] = (ycs - (grid_y_idxs.float() + 0.5)) / (self.window_size / 2.)
        gt_face[batch_idxs, 3, grid_y_idxs, grid_x_idxs] = (x2_gts - x1_gts) / nw  # 归一化尺寸
        gt_face[batch_idxs, 4, grid_y_idxs, grid_x_idxs] = (y2_gts - y1_gts) / nh  # 归一化尺寸

        return gt_face
