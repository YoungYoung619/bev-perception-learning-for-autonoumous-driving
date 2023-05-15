from copy import deepcopy
import numpy as np

def load_boxes3d_groundtruth(meta, fill_key='boxes3d'):
    anns = meta['ann_info']['anns']
    boxes3d = []
    boxes3d_standard = 'LIDAR_TOP'
    for ann in anns:
        size = ann['size']
        category_name = ann['category_name']
        visibility = ann['visibility']
        translation = ann[boxes3d_standard]['ego_frame']['translation']
        rotation = ann[boxes3d_standard]['ego_frame']['rotation']
        velocity = ann[boxes3d_standard]['ego_frame']['velocity']
        n_pts = ann['num_pts']  # lidar+radar点数
        box3d = {
            'size': size,
            'translation': translation,
            'rotation': rotation,
            'velocity': velocity,
            'category_name': category_name,
            'visibility': visibility,
            'num_pts': n_pts,
        }
        boxes3d.append(box3d)
    meta[fill_key] = boxes3d
    return meta

def load_map_groundtruth(meta, ploy_names, line_names):
    ann = meta['ann_info']
    for ploy_name in ploy_names:
        if ploy_name in ann:
            meta[ploy_name] = deepcopy(ann[ploy_name])
    for line_name in line_names:
        if line_name in ann:
            meta[line_name] = deepcopy(ann[line_name])
    return meta