import cv2
import numpy as np
import tensorflow as tf
import hashlib
import os
from dataset.DataSetRead import DataSetRead
from dataset.dataset_util import IoU


# coarse convert
def convert_ellipse_2_rectangle(data_lines):
    gt_bboxes = []
    for data_line in data_lines:
        print(data_line)
        major_axis_radius, minor_axis_radius, angle, center_x, center_y, _ = map(float, data_line.strip().split())
        gt_bboxes.append([center_x-minor_axis_radius, center_y-major_axis_radius, 2*minor_axis_radius, 2*major_axis_radius])
    return np.array(gt_bboxes, dtype=np.int)


class FDDB(DataSetRead):
    def __init__(self):
        super(FDDB, self).__init__()
        self.dataset_dir = '/home/xuhai/disk2/ykfan/data/FDDB'

    def _read_annotations(self, data_type, limit_count=-1):
        assert isinstance(data_type, int), 'data type should be int type.'
        annotation_list = list()
        annotation_txt = '{}/FDDB-folds/FDDB-fold-{:0>2d}-ellipseList.txt'.format(self.dataset_dir, data_type)
        annotations = open(annotation_txt).readlines()
        index = 0
        while index < len(annotations):
            if 0 < limit_count < len(annotation_list):
                break

            annotation_name = annotations[index].strip()
            bbox_len = int(annotations[index + 1])

            annotation_val = dict()
            annotation_val['file_name'] = annotation_name
            annotation_val['id'] = annotation_name
            annotation_val['dataset'] = 'fddb'
            annotation_val['data'] = convert_ellipse_2_rectangle(annotations[index + 2:index + 2 + bbox_len])
            annotation_val['img_dir'] = self.dataset_dir
            annotation_list.append(annotation_val)
            index += bbox_len + 2

        return annotation_list

    def decode_annotation_val(self, annotation_val):
        full_path = '{}/{}.jpg'.format(annotation_val['img_dir'], annotation_val['file_name'])
        image = cv2.imread(full_path)
        assert image is not None, 'no image {} exists!'.format(full_path)
        image_height, image_width, _ = image.shape
        with tf.gfile.GFile(full_path, 'rb') as fid:
            encoded_jpg = fid.read()
        key = hashlib.sha256(encoded_jpg).hexdigest()

        xmin = []
        xmax = []
        ymin = []
        ymax = []
        is_crowd = []
        category_names = []
        category_ids = []
        area = []
        num_annotations_skipped = 0
        for object_annotation in annotation_val['data']:
            x, y, width, height = object_annotation
            # (x, y, width, height) = tuple(object_annotations['bbox'])
            if width <= 0 or height <= 0:
                tf.logging.debug('skip for width {}, height {}'.format(width, height))
                num_annotations_skipped += 1
                continue
            if x + width > image_width or y + height > image_height:
                tf.logging.debug('skip for beyond image range.')
                num_annotations_skipped += 1
                continue
            xmin.append(float(x) / image_width)
            xmax.append(float(x + width) / image_width)
            ymin.append(float(y) / image_height)
            ymax.append(float(y + height) / image_height)
            category_id = 0
            category_names.append(self.category_index[category_id]['name'].encode('utf8'))
            cv2.rectangle(image, (x, y), (x + width, y + height), self.category_index[category_id]['color'])
            # unused label
            category_ids.append(category_id)
            is_crowd.append(0)
            area.append(0.0)

        annotation_val['image_height'] = image_height
        annotation_val['image_width'] = image_width
        annotation_val['encoded_jpg'] = encoded_jpg
        annotation_val['key'] = key
        annotation_val['xmin'] = xmin
        annotation_val['xmax'] = xmax
        annotation_val['ymin'] = ymin
        annotation_val['ymax'] = ymax
        annotation_val['is_crowd'] = is_crowd
        annotation_val['category_names'] = category_names
        annotation_val['category_ids'] = category_ids
        annotation_val['area'] = area
        annotation_val['num_annotations_skipped'] = num_annotations_skipped
        annotation_val['verify_img'] = image
        return annotation_val

    def evaluate(self, detect_estimator, iou_threshold=0.3, verify_dir=None, limit_count=-1):
        annotation_list = self.read_annotations(range(1, 11), limit_count=limit_count)
        total_precision_sum = 0
        total_precision_tp = 0
        total_recall_sum = 0
        total_recall_tp = 0

        for index, annotation in enumerate(annotation_list):
            precision_tp = 0
            image_path = '{}/{}.jpg'.format(annotation['img_dir'], annotation['file_name'])
            img = cv2.imread(image_path)
            assert img is not None, '{} should not be None'.format(annotation['file_name'])
            pred_bboxes = detect_estimator(image_path)
            gt_bboxes = annotation['data']
            gt_mask = np.zeros([len(gt_bboxes)])

            for pred_bbox in pred_bboxes:
                iou_vals = IoU(pred_bbox, gt_bboxes)
                pred_bbox = [int(_) for _ in pred_bbox]
                # print iou_vals
                max_index = np.argmax(iou_vals)
                if iou_vals[max_index] >= iou_threshold:
                    gt_mask[max_index] = 1
                    precision_tp += 1
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3]), (0, 255, 0))
                else:
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3]), (0, 0, 255))

            for gt_bbox in gt_bboxes:
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (255, 0, 0))

            recall_tp = np.sum(gt_mask)

            total_precision_sum += len(pred_bboxes)
            total_precision_tp += precision_tp
            total_recall_sum += len(gt_bboxes)
            total_recall_tp += recall_tp

            if index % 100 == 0 or index == len(annotation_list) - 1:
                if not os.path.exists(verify_dir):
                    os.mkdir(verify_dir)
                cv2.imwrite('{}/verify_{}.jpg'.format(verify_dir, index), img)
                print('image precision {:.1f} %, recall {:.1f} %'.format(
                    float(precision_tp) * 100.0 / (float(len(pred_bboxes)) + 0.000001),
                    float(recall_tp) * 100.0 / (float(len(gt_bboxes)) + 0.000001)
                ))
                print( '{}/{} total precision {:.1f} %, recall {:.1f} %'.format(
                    index,
                    len(annotation_list),
                    float(total_precision_tp) * 100.0 / float(total_precision_sum),
                    float(total_recall_tp) * 100.0 / float(total_recall_sum)
                ))


if __name__ == '__main__':
    pass
    # def detector_estimator(img):
    #     return {'bboxes': np.array([[0, 0, 20, 20], [0, 0, 30, 30],])}
    #
    # evaluate(detect_estimator=detector_estimator, limit_count=20)
