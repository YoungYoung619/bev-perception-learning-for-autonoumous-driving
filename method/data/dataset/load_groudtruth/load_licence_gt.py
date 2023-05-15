import numpy as np
from method.data.transform.warp import warp_boxes

def load_two_stage_licence_gt(meta, matrix, dst_shape):
    gt_licence_exist = False
    gt_attach_licence_exist = False
    gt_licence = [0., 0., 0., 0.]
    gt_attach_licence = [0., 0., 0., 0.]
    img_info = meta['img_info']
    if img_info['task_name'] == 'licence':
        if 'licence_ann' in img_info:
            # 关键点检测使用这个
            # gt_licence_exist = img_info['licence_ann']['kps']['exist']
            # gt_licence = [img_info['licence_ann']['kps']['ann']['right_bottom']['x'],
            #               img_info['licence_ann']['kps']['ann']['right_bottom']['y'],
            #               img_info['licence_ann']['kps']['ann']['left_bottom']['x'],
            #               img_info['licence_ann']['kps']['ann']['left_bottom']['y'],
            #               img_info['licence_ann']['kps']['ann']['left_top']['x'],
            #               img_info['licence_ann']['kps']['ann']['left_top']['y'],
            #               img_info['licence_ann']['kps']['ann']['right_top']['x'],
            #               img_info['licence_ann']['kps']['ann']['right_top']['y']]

            # 2d框检测使用这个
            gt_licence_exist = img_info['licence_ann']['box']['exist']
            gt_licence = [
                img_info['licence_ann']['box']['ann']['x1'],
                img_info['licence_ann']['box']['ann']['y1'],
                img_info['licence_ann']['box']['ann']['x2'],
                img_info['licence_ann']['box']['ann']['y2'],
            ]

            # 补充标签，避免crash
            if 'blur' not in img_info['licence_ann']['box']:
                img_info['licence_ann']['box']['blur'] = False
            if 'occupation' not in img_info['licence_ann']['box']:
                img_info['licence_ann']['box']['occupation'] = False
            if 'kps' not in img_info['licence_ann']:
                img_info['licence_ann']['kps'] = {
                    'exist': False,
                    'ann': {
                        'right_bottom': {'x': 0, 'y': 0},
                        'left_bottom': {'x': 0, 'y': 0},
                        'left_top': {'x': 0, 'y': 0},
                        'right_top': {'x': 0, 'y': 0},
                    }
                }
        else:
            # 补充标签，避免叠加batch时crash
            img_info['licence_ann'] = {}
            img_info['licence_ann']['box'] = {
                'exist': False,
                'blur': False,
                'occupation': False,
                'ann': {
                    'x1': 0.,
                    'y1': 0.,
                    'x2': 0.,
                    'y2': 0.
                }
            }
            img_info['licence_ann']['kps'] = {
                'exist': False,
                'ann': {
                    'right_bottom': {'x': 0, 'y': 0},
                    'left_bottom': {'x': 0, 'y': 0},
                    'left_top': {'x': 0, 'y': 0},
                    'right_top': {'x': 0, 'y': 0},
                }
            }

        if 'attach_licence_ann' in img_info:
            gt_attach_licence_exist = img_info['attach_licence_ann']['box']['exist']
            gt_attach_licence = [
                img_info['attach_licence_ann']['box']['ann']['x1'],
                img_info['attach_licence_ann']['box']['ann']['y1'],
                img_info['attach_licence_ann']['box']['ann']['x2'],
                img_info['attach_licence_ann']['box']['ann']['y2'],
            ]
        else:
            # 补充标签，避免叠加batch时crash
            img_info['attach_licence_ann'] = {}
            img_info['attach_licence_ann']['box'] = {
                'exist': False,
                'blur': False,
                'occupation': False,
                'ann': {
                    'x1': 0.,
                    'y1': 0.,
                    'x2': 0.,
                    'y2': 0.
                }
            }

        if 'attach_licence_ann_1' in img_info:
            gt_attach_licence_exist = img_info['attach_licence_ann_1']['box']['exist']
            gt_attach_licence = [
                img_info['attach_licence_ann_1']['box']['ann']['x1'],
                img_info['attach_licence_ann_1']['box']['ann']['y1'],
                img_info['attach_licence_ann_1']['box']['ann']['x2'],
                img_info['attach_licence_ann_1']['box']['ann']['y2'],
            ]
        else:
            # 补充标签，避免叠加batch时crash
            img_info['attach_licence_ann_1'] = {}
            img_info['attach_licence_ann_1']['box'] = {
                'exist': False,
                'blur': False,
                'occupation': False,
                'ann': {
                    'x1': 0.,
                    'y1': 0.,
                    'x2': 0.,
                    'y2': 0.
                }
            }

    gt_meta = {}
    gt_licence_exist, gt_licence = ([gt_licence_exist, gt_attach_licence_exist], [gt_licence, gt_attach_licence])
    gt_meta['gt_licence'] = [gt_licence_exist, gt_licence]
    meta.update(gt_meta)

    # 车牌2d目标检测时的变换
    if 'gt_licence' in meta and (meta['gt_licence'][0][0] or meta['gt_licence'][0][1]):
        boxes = np.array([meta['gt_licence'][1]], dtype=np.float32)[0]
        boxes = warp_boxes(boxes, matrix, dst_shape[1], dst_shape[0])
        meta['gt_licence'][1] = boxes.tolist()
        for i in range(len(boxes)):
            box = boxes[i]
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area < 10:
                meta['gt_licence'][0][i] = False
                meta['gt_licence'][1][i] = [0., 0., 0., 0.]

    # # ----- debug ------#
    # import cv2
    # def _de_normalize(img, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395]):
    #     mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3) / 255
    #     std = np.array(std, dtype=np.float32).reshape(1, 1, 3) / 255
    #     img = img * std + mean
    #     img = (img * 255).astype(np.uint8)
    #     return img
    #
    # if 'gt_licence' in meta and meta['gt_licence'][0] != []:
    #     if meta['gt_licence'][0]:
    #         img = meta['img'].copy()
    #         img = _de_normalize(img)
    #         boxes = np.array(meta['gt_licence'][1], dtype=np.float32)
    #         print(boxes)
    #         for i, box in enumerate(boxes):
    #             color = (0, 0, 255) if i == 0 else (0, 255, 0)
    #             cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness=1)
    #     # cv2.imshow("raw", raw_img)
    #     cv2.imshow("debug", img)
    #     cv2.waitKey(0)
    # # ---- debug ------#

    return meta
