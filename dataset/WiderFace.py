import cv2
import tensorflow as tf
import hashlib
import os
from dataset.DataSetRead import DataSetRead

class WiderFace(DataSetRead):
    def __init__(self):
        super(WiderFace, self).__init__()
        self.dataset_dir = '/home/xuhai/disk2/ykfan/data/WiderFace'

    def _read_annotations(self, data_type, limit_count=-1):
        assert data_type in ['train', 'val', ], 'annotatin type for wider face should be train or val.'
        annotation_txt = '{}/wider_face_split/wider_face_{}_bbx_gt.txt'.format(self.dataset_dir, data_type)
        img_dir = '{}/WIDER_{}/images'.format(self.dataset_dir, data_type)
        annotation_list = list()

        annotations = open(annotation_txt).readlines()
        index = 0
        while index < len(annotations):
            annotation_name = annotations[index].strip()
            bbox_len = int(annotations[index + 1])

            annotation_val = dict()
            annotation_val['file_name'] = annotation_name
            annotation_val['id'] = annotation_name
            annotation_val['dataset'] = 'WiderFace'
            annotation_val['data'] = annotations[index + 2:index + 2 + bbox_len]
            annotation_val['img_dir'] = img_dir
            annotation_list.append(annotation_val)
            index += bbox_len + 2

            if 0 < limit_count < len(annotation_list):
                break
        tf.logging.info('wider face {} dataset, with size {}'.format(data_type, len(annotation_list)))
        return annotation_list


    def decode_annotation_val(self, annotation_val):
        full_path = os.path.join(annotation_val['img_dir'], annotation_val['file_name'])
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
        for object_annotations in annotation_val['data']:
            x, y, width, height = list(map(int, object_annotations.split()))[:4]
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


if __name__ == '__main__':
    pass
    # train_txt = '{}/wider_face_split/wider_face_train_bbx_gt.txt'.format(dataset_dir)
    #
    # train_annotations = open(train_txt).readlines()
    #
    # index = 0
    # while index < len(train_annotations):
    #     img_path = train_annotations[index].strip()
    #     img_path = '{}/WIDER_train/images/{}'.format(dataset_dir, img_path)
    #     print(img_path)
    #     img = cv2.imread(img_path)
    #     pose_atypical_num = 0
    #
    #     bbox_len = int(train_annotations[index + 1])
    #     for bbox_index in range(bbox_len):
    #         x1, y1, w, h, _, _, __, _, _, pose = map(int, train_annotations[index + 2 + bbox_index].split())
    #         if pose == 1 and w > 20 and h > 20:
    #             pose_atypical_num += 1
    #             cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0))
    #
    #     if pose_atypical_num > 0:
    #         cv2.imshow('wider face', img)
    #         cv2.waitKey(0)
    #
    #     index += bbox_len + 2


