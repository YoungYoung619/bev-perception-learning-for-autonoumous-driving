

def load_desensitive_cls_gt(meta, matrix, dst_shape):
    gt_desensitive_cls = []
    img_info = meta['img_info']
    if 'desensitive_cls_ann' in img_info:
        gt_desensitive_cls = img_info['desensitive_cls_ann']['label']

    meta['gt_desensitive_cls'] = gt_desensitive_cls
    return meta