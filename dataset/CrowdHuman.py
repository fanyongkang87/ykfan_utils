import logging
import cv2
import tensorflow as tf
import hashlib
import json
import os
import dataset.dataset_util as dataset_util

# {"ID": "284193,faa9000f2678b5e", "gtboxes":
#     [{"tag": "person", "hbox": [123, 129, 63, 64], "head_attr": {"ignore": 0, "occ": 1, "unsure": 0}, "fbox": [61, 123, 191, 453], "vbox": [62, 126, 154, 446], "extra": {"box_id": 0, "occ": 1}},
#      {"tag": "person", "hbox": [214, 97, 58, 74], "head_attr": {"ignore": 0, "occ": 1, "unsure": 0}, "fbox": [165, 95, 187, 494], "vbox": [175, 95, 140, 487], "extra": {"box_id": 1, "occ": 1}},
#      {"tag": "person", "hbox": [318, 109, 58, 68], "head_attr": {"ignore": 0, "occ": 1, "unsure": 0}, "fbox": [236, 104, 195, 493], "vbox": [260, 106, 170, 487], "extra": {"box_id": 2, "occ": 1}},
#      {"tag": "person", "hbox": [486, 119, 61, 74], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}, "fbox": [452, 110, 169, 508], "vbox": [455, 113, 141, 501], "extra": {"box_id": 3, "occ": 1}},
#      {"tag": "person", "hbox": [559, 105, 53, 57], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}, "fbox": [520, 95, 163, 381], "vbox": [553, 98, 70, 118], "extra": {"box_id": 4, "occ": 1}},
#      {"tag": "person", "hbox": [596, 40, 72, 83], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}, "fbox": [546, 39, 202, 594], "vbox": [556, 39, 171, 588], "extra": {"box_id": 5, "occ": 1}},
#      {"tag": "person", "hbox": [731, 139, 69, 83], "head_attr": {"ignore": 0, "occ": 0, "unsure": 0}, "fbox": [661, 132, 183, 510], "vbox": [661, 132, 183, 510], "extra": {"box_id": 6, "occ": 0}}]}

dataset_dir = '/home/ykfan/data/CrowdHuman'
category_index = [{'name':'face', 'color':(255, 0, 0)}, {'name':'head', 'color':(0, 255, 0)}, {'name':'shoulder', 'color':(0, 0, 255)},]


def box_pading(box, vbox):
    if box[0] < vbox[0]:
        box[0] = vbox[0]
    if box[1] < vbox[1]:
        box[1] = vbox[1]
    if box[2] > vbox[0]+vbox[2]:
        box[2] = vbox[0]+vbox[2]
    if box[3] > vbox[1]+vbox[3]:
        box[3] = vbox[1]+vbox[3]
    return list(map(int, box))


# return format is [is_use, shoulder xmin, ymin, xmax, ymax, face xmin ymin, xmax, ymax]
def box_shoulder(hbox, vbox):
    logging.debug('hbox {}'.format(hbox))
    logging.debug('vbox {}'.format(vbox))
    hbox = list(map(float, hbox))
    vbox = list(map(float, vbox))
    hbox_w = hbox[2]
    hbox_h = hbox[3]

    shoulder_box = [hbox[0]-0.5*hbox_w, hbox[1], hbox[0]+1.5*hbox_w, hbox[1]+2.0*hbox_h]
    return box_pading(shoulder_box, vbox)

def get_shoulder_boxes(annotation_val, image_width, image_height):
    gt_shouder_boxes = []
    for object_annotations in annotation_val['data']:
        # print gtboxes[index]['tag']
        if not object_annotations['tag'] == 'person':
            continue
        head_box = list(map(int, object_annotations['hbox']))
        x, y, width, height = head_box
        if width <= 0 or height <= 0:
            continue
        if x + width > image_width or y + height > image_height:
            continue

        virt_box = list(map(int, object_annotations['vbox']))
        shoulder_box = box_shoulder(head_box, virt_box)
        gt_shouder_boxes.append(shoulder_box)
    return gt_shouder_boxes

def read_annotations(annotation_type, limit_count = -1):
    if isinstance(annotation_type, list):
        annotations = list()
        for type_val in annotation_type:
            annotations.extend(read_annotations(type_val, limit_count))
        return annotations

    assert annotation_type in ['train', 'val'], 'annotation type should be in type train,val.'
    annotation_txt = 'annotation_{}.odgt'.format(annotation_type)
    annotation_lines = open(os.path.join(dataset_dir, annotation_txt)).readlines()
    img_dir = '{}/Images'.format(dataset_dir)

    annotation_list = []
    for annotation in annotation_lines:
        # print annotation
        annotation = json.loads(annotation)
        annotation_val = dict()

        annotation_val['id'] = annotation['ID']
        annotation_val['file_name'] = '{}.jpg'.format(annotation_val['id'])
        annotation_val['dataset'] = 'CrowdHuman'
        annotation_val['data'] = annotation['gtboxes']
        annotation_val['img_dir'] = img_dir
        annotation_list.append(annotation_val)

        if 0 < limit_count < len(annotation_list):
            break
    tf.logging.info('crowd human {} dataset, with size {}'.format(annotation_type, len(annotation_list)))
    return annotation_list

def decode_annotation(annotation_val):
    full_path = os.path.join(annotation_val['img_dir'], annotation_val['file_name'])
    image = cv2.imread(full_path)
    assert image is not None, 'no image {} exists!'.format(full_path)
    image_height, image_width, _ = image.shape

    gt_head_boxes = []
    gt_shoulder_boxes = []
    for object_annotations in annotation_val['data']:
        # print gtboxes[index]['tag']
        if not object_annotations['tag'] == 'person':
            continue
        head_box = list(map(int, object_annotations['hbox']))
        x, y, width, height = head_box
        if width <= 0 or height <= 0:
            tf.logging.debug('skip for width {}, height {}'.format(width, height))
            continue
        if x + width > image_width or y + height > image_height:
            tf.logging.debug('skip for beyond image range.')
            continue
        virt_box = list(map(int, object_annotations['vbox']))
        shoulder_box = box_shoulder(head_box, virt_box)
        # now gt boxes are all [x1, y1, x2, y2] format
        gt_head_boxes.append([x, y, x+width, y+height])
        gt_shoulder_boxes.append(shoulder_box)

    annotation_val['gt_head_boxes'] = gt_head_boxes
    annotation_val['gt_shoulder_boxes'] = gt_shoulder_boxes
    annotation_val['image'] = image
    return annotation_val


def create_tf_example(annotation_val, verify_dict=None):
    filename = annotation_val['file_name']
    image_id = annotation_val['id']

    full_path = os.path.join(annotation_val['img_dir'], filename)
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
        # print gtboxes[index]['tag']
        if not object_annotations['tag'] == 'person':
            continue

        head_box = list(map(int, object_annotations['hbox']))
        x, y, width, height = head_box
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
        category_id = 1
        cv2.rectangle(image, (x,y), (x+width, y+height), category_index[category_id]['color'])
        category_names.append(category_index[category_id]['name'].encode('utf8'))
        #unused label
        is_crowd.append(0)
        area.append(0.0)

        virt_box = list(map(int, object_annotations['vbox']))
        shoulder_box = box_shoulder(head_box, virt_box)
        x1, y1, x2, y2 = shoulder_box

        xmin.append(float(x1) / image_width)
        xmax.append(float(x2) / image_width)
        ymin.append(float(y1) / image_height)
        ymax.append(float(y2) / image_height)
        category_id = 2
        cv2.rectangle(image, (x1,y1), (x2, y2), category_index[category_id]['color'])
        category_names.append(category_index[category_id]['name'].encode('utf8'))
        #unused label
        is_crowd.append(0)
        area.append(0.0)

    if verify_dict is not None:
        cv2.imwrite(verify_dict['save_path'], image)

    feature_dict = {
        'image/height':
            dataset_util.int64_feature(image_height),
        'image/width':
            dataset_util.int64_feature(image_width),
        'image/filename':
            dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id':
            dataset_util.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded':
            dataset_util.bytes_feature(encoded_jpg),
        'image/format':
            dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymax),
        'image/object/class/text':
            dataset_util.bytes_list_feature(category_names),
        'image/object/is_crowd':
            dataset_util.int64_list_feature(is_crowd),
        'image/object/area':
            dataset_util.float_list_feature(area),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return key, example, num_annotations_skipped

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    verify_dir = 'verify'
    count = 0
    os.mkdir(verify_dir)
    for val in read_annotations(['train', 'val']):
        count += 1
        if count % 1000 != 0:
            continue

        val = decode_annotation(val)
        img = val['image']
        for box in val['gt_head_boxes']:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))
        for box in val['gt_shoulder_boxes']:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255))

        cv2.imwrite('{}/out_{}.jpg'.format(verify_dir, count), img)
        logging.info('verify image {}'.format(count))


