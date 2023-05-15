import numpy as np
from method.data.transform.warp import warp_boxes


def load_det_gt(img_info, matrix, dst_shape):
    gt_bboxes = []
    gt_labels = []

    # parase detection label
    det_ann = img_info['det_ann']
    for bbox in det_ann:
        gt_bboxes.append(bbox['bbox'])
        gt_labels.append(bbox['label'])
    if gt_bboxes:
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
    else:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)

    gt_bboxes = warp_boxes(gt_bboxes, matrix, dst_shape[0], dst_shape[1])
    return {'gt_bboxes': gt_bboxes, 'gt_labels': gt_labels}


def load_face_gt(img_info, matrix, dst_shape):
    gt_bboxes = []
    gt_labels = []

    # parase detection label
    face_det_ann = img_info['face_det_ann']
    for bbox in face_det_ann:
        gt_bboxes.append(bbox['bbox'])
        gt_labels.append(bbox['label'])
    if gt_bboxes:
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
    else:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)

    gt_bboxes = warp_boxes(gt_bboxes, matrix, dst_shape[0], dst_shape[1])
    return {'gt_face_bboxes': gt_bboxes, 'gt_face_labels': gt_labels}
