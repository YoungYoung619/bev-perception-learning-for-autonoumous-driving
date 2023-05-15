# two stage face model ground truth loader
import numpy as np
from method.data.transform.warp import warp_boxes

def load_two_stage_face_gt(meta, matrix, dst_shape):
    gt_face_exist = []
    gt_face_boxes = []  # x1 y1 x2 y2
    img_info = meta['img_info']
    if img_info['task_name'] == 'face':
        for face_ann in img_info['face_ann']['box']:
            x1, y1, x2, y2 = face_ann['ann']['x1'], face_ann['ann']['y1'], face_ann['ann']['x2'], face_ann['ann'][
                'y2']
            gt_face_boxes.append([x1, y1, x2, y2])
            gt_face_exist.append(True)

        if 'face_ann_val' not in img_info:
            img_info['face_ann_val'] = {'box': None}
            # 修改内容，以支持online validation
            new_box_dict = {
                'blur': [],
                'occupation': [],
                'visible': [],
                'pose': [],
                'box': [],
            }
            for face_ann_box in img_info['face_ann']['box']:
                new_box_dict['blur'].append(face_ann_box['blur'])
                new_box_dict['occupation'].append(face_ann_box['occupation'])
                new_box_dict['visible'].append(face_ann_box['visible'])
                new_box_dict['pose'].append(face_ann_box['pose'])
                new_box_dict['box'].append([face_ann_box['ann']['x1'],
                                            face_ann_box['ann']['y1'],
                                            face_ann_box['ann']['x2'],
                                            face_ann_box['ann']['y2']])
            for key, val in new_box_dict.items():
                new_box_dict[key] = np.array(val)
            img_info['face_ann_val']['box'] = new_box_dict

    gt_meta = {}
    gt_meta['gt_face'] = [gt_face_exist, gt_face_boxes]
    meta.update(gt_meta)

    if 'gt_face' in meta and meta['gt_face']:
        boxes = np.array(meta['gt_face'][1], dtype=np.float32)
        boxes = warp_boxes(boxes, matrix, dst_shape[1], dst_shape[0])
        meta['gt_face'][1] = boxes.tolist()
        for i in range(len(boxes)):
            box = boxes[i]
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area < 10:
                meta['gt_face'][0][i] = False
                meta['gt_face'][1][i] = [0., 0., 0., 0.]
        max_n = 10
        rest_n = max_n - len(meta['gt_face'][0])
        if rest_n > 0:
            meta['gt_face'][1] += [[0., 0., 0., 0.]] * rest_n
            meta['gt_face'][0] += [False] * rest_n
        elif rest_n < 0:
            meta['gt_face'][1] = meta['gt_face'][1][:max_n]
            meta['gt_face'][0] = meta['gt_face'][0][:max_n]

    # # ----- debug ------#
    # def _de_normalize(img, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395]):
    #     mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3) / 255
    #     std = np.array(std, dtype=np.float32).reshape(1, 1, 3) / 255
    #     img = img * std + mean
    #     img = (img * 255).astype(np.uint8)
    #     return img
    #
    # import cv2
    # if 'gt_face' in meta and meta['gt_face'][0] != []:
    #     if meta['gt_face'][0]:
    #         img = meta['img'].copy()
    #         img = _de_normalize(img)
    #         boxes = np.array(meta['gt_face'][1], dtype=np.float32)
    #         print(boxes)
    #         for i, box in enumerate(boxes):
    #             color = (0, 0, 255) if i == 0 else (0, 255, 0)
    #             cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness=1)
    #     # cv2.imshow("raw", raw_img)
    #     cv2.imshow("debug", img)
    #     cv2.waitKey(0)
    return meta
