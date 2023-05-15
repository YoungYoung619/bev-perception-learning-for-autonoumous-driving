import config
import pickle
import json
import numpy as np


def save_gt_coco():
    """测试用"""
    with open(config.val_gt_path, 'rb') as f:
        gt_dict = pickle.load(f)

    ret = {'images': [], 'annotations': [], "categories": [{'name': 'vehicle', 'id': 0}, {'name': 'person', 'id': 1}]}

    n_person = 0
    n_vehicle = 0
    for key in gt_dict.keys():
        print('processing %s...' % (key))
        bboxes = gt_dict[key]

        image_info = {'file_name': key,
                      'id': key}
        ret['images'].append(image_info)

        for box in bboxes:
            xmin, ymin, xmax, ymax, label, _ = box

            if label == 0:
                n_vehicle += 1
            elif label == 1:
                n_person += 1
            else:
                raise ValueError

            x = int(xmin * 1280)
            y = int(ymin * 720)
            w = int((xmax - xmin) * 1280)
            h = int((ymax - ymin) * 720)
            bbox = [x, y, w, h]
            ann = {'image_id': key,
                   'id': int(len(ret['annotations']) + 1),
                   'category_id': label,
                   'bbox': bbox,
                   'area': bbox[2] * bbox[3],
                   'ignore': 0,
                   'iscrowd': 0
                   }
            ret['annotations'].append(ann)

    out_path = './annotations/val_gt_coco.json'
    json.dump(ret, open(out_path, 'w'))
    print('success... n_vehicle:%d  n_person:%d' % (n_vehicle, n_person))
    pass


def save_raw_pred_2_coco():
    """测试用"""
    with open('./result_dict_raw.pkl', 'rb') as file:
        pred_res = pickle.load(file)

    detections = []
    for key in pred_res.keys():
        print('processing %s...' % (key))
        bboxes = pred_res[key]

        for bbox in bboxes:
            score, x1, y1, x2, y2, cls_id = bbox

            x = int(x1 * 1280)
            y = int(y1 * 720)
            w = int((x2 - x1) * 1280)
            h = int((y2 - y1) * 720)

            detection = {
                "image_id": int(key),
                "category_id": int(cls_id),
                "bbox": [x, y, w, h],
                "score": float("{:.2f}".format(score))
            }
            detections.append(detection)

    with open('./prediction/results_pred.json', 'w') as file:
        json.dump(detections, file)


def convert_gt_dict_2_coco(gt_dict, width=720, height=1280):
    """将原来gt_dict转换成coco格式"""
    ## [{'name': 'vehicle', 'id': 0}, {'name': 'person', 'id': 1}]
    c_list = []
    for idx, name in enumerate(config.class_list):
        c_list.append({'name': name, 'id': idx})

    ret = {'images': [], 'annotations': [], "categories": c_list}

    for key in gt_dict.keys():
        # print('processing %s...' % (key))
        bboxes = gt_dict[key]

        image_info = {'file_name': key,
                      'id': key}
        ret['images'].append(image_info)

        for box in bboxes:
            xmin, ymin, xmax, ymax, label, _ = box

            x = int(xmin * height)
            y = int(ymin * width)
            w = int((xmax - xmin) * height)
            h = int((ymax - ymin) * width)
            bbox = [x, y, w, h]
            ann = {'image_id': key,
                   'id': int(len(ret['annotations']) + 1),
                   'category_id': label,
                   'bbox': bbox,
                   'area': bbox[2] * bbox[3],
                   'ignore': 0,
                   'iscrowd': 0
                   }
            ret['annotations'].append(ann)

    return ret


def convert_res_dict_2_coco(results_dict, width=720, height=1280):
    """转换原来的resultsd_dict到coco的格式"""
    detections = []
    for key in results_dict.keys():
        # print('processing %s...' % (key))
        bboxes = results_dict[key]

        for bbox in bboxes:
            score, x1, y1, x2, y2, cls_id = bbox

            x = int(x1 * height)
            y = int(y1 * width)
            w = int((x2 - x1) * height)
            h = int((y2 - y1) * width)

            detection = {
                "image_id": int(key),
                "category_id": int(cls_id),
                "bbox": [x, y, w, h],
                "score": float("{:.2f}".format(score))
            }
            detections.append(detection)

    return detections


def convert_to_coco_format(task, res, cls_configs=None):
    """
    Args:
        task: 任务类别
        res: 指定task下的检测结果
        cls_configs: 所有任务下的类别配置表
    Returns:

    """
    cls_configs = {
        'face': ['face'],
        'moving_obstacle': ['person', 'vehicle']
    } if cls_configs is None else cls_configs
    c_list = []
    for idx, name in enumerate(cls_configs[task]):
        c_list.append({'name': name, 'id': idx})

    filenames = list(res.keys())
    detections = []
    ret = {'images': [], 'annotations': [], "categories": c_list}
    for img_id, filename in enumerate(filenames):
        res_one_img = res[filename]['pred']
        gt_one_img = res[filename]['gt']

        for cls_idx, bboxes in res_one_img.items():
            cls_idx = int(cls_idx)
            for box in bboxes:
                x, y, w, h, score = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1]), box[4]
                detection = {
                    "image_id": img_id,
                    "category_id": cls_idx,
                    "bbox": [x, y, w, h],
                    "score": float("{:.2f}".format(score))
                }
                detections.append(detection)

        image_info = {'file_name': filename,
                      'id': img_id}
        ret['images'].append(image_info)
        for cls_idx, bboxes in gt_one_img.items():
            cls_idx = int(cls_idx)
            for box in bboxes:
                x, y, w, h = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])
                ann = {'image_id': img_id,
                       'id': int(len(ret['annotations']) + 1),
                       'category_id': cls_idx,
                       'bbox': [x, y, w, h],
                       'area': w * h,
                       'ignore': 0,
                       'iscrowd': 0
                       }
                ret['annotations'].append(ann)
    return ret, detections

if __name__ == '__main__':
    pass
