import contextlib2
import tensorflow as tf
from tqdm import tqdm
import cv2
import dataset.dataset_util as dataset_util
from common_utils.dir_utils import clear_dir


class DataSetRead(object):
    def __init__(self):
        print('data set read init.')
        self.category_index = [{'name':'face', 'color':(255, 0, 0)}, {'name':'head', 'color':(0, 255, 0)}, {'name':'shoulder', 'color':(0, 0, 255)},]

    def _read_annotations(self, data_type, limit_count=-1):
        raise NotImplementedError('virtual function')

    def read_annotations(self, data_type_list, limit_count=-1):
        assert type(data_type_list) == list, 'data type list should be type list'
        total_list = list()
        for data_type in data_type_list:
            annotation_list = self._read_annotations(data_type, limit_count)
            total_list.extend(annotation_list)
        return total_list

    def decode_annotation_val(self, annotation_val):
        raise NotImplementedError('virtual function')

    def create_tf_example(self, annotation_val, verify_dict=None, feature_spec_dict=None, idx=0):
        annotation_val = self.decode_annotation_val(annotation_val)
        if verify_dict is not None and idx%verify_dict['verify_count']==0:
            cv2.imwrite('{}/verify_{}.jpg'.format(verify_dict['verify_dir'], idx), annotation_val['verify_img'])

        feature_dict = {
            'image/height':
                dataset_util.int64_feature(annotation_val['image_height']),
            'image/width':
                dataset_util.int64_feature(annotation_val['image_width']),
            'image/filename':
                dataset_util.bytes_feature(annotation_val['file_name'].encode('utf8')),
            'image/source_id':
                dataset_util.bytes_feature(str(annotation_val['id']).encode('utf8')),
            'image/key/sha256':
                dataset_util.bytes_feature(annotation_val['key'].encode('utf8')),
            'image/encoded':
                dataset_util.bytes_feature(annotation_val['encoded_jpg']),
            'image/format':
                dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin':
                dataset_util.float_list_feature(annotation_val['xmin']),
            'image/object/bbox/xmax':
                dataset_util.float_list_feature(annotation_val['xmax']),
            'image/object/bbox/ymin':
                dataset_util.float_list_feature(annotation_val['ymin']),
            'image/object/bbox/ymax':
                dataset_util.float_list_feature(annotation_val['ymax']),
            'image/object/class/text':
                dataset_util.bytes_list_feature(annotation_val['category_names']),
            'image/object/is_crowd':
                dataset_util.int64_list_feature(annotation_val['is_crowd']),
            'image/object/area':
                dataset_util.float_list_feature(annotation_val['area']),
        }

        if feature_spec_dict:
            spec_feature_dict = dict()
            for spec_key, spec_val in feature_spec_dict.items():
                spec_feature_dict[spec_key] = feature_dict[spec_val]
            feature_dict = spec_feature_dict

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return annotation_val['key'], example, annotation_val['num_annotations_skipped']

    def create_tf_records(self, data_type_list, output_path, num_shards, verify_dict=None, feature_spec_dict=None):
        '''
        :param data_type_list:
        :param output_path:
        :param num_shards:
        :param verify_dict: {'verify_count': 1000, 'verify_dir': 'data/verify_train'},
        :param feature_spec_dict:
        :return:
        '''

        clear_dir(output_path)
        if verify_dict is not None:
            clear_dir(verify_dict['verify_dir'])

        annotations_list = self.read_annotations(data_type_list, limit_count=100)
        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tfrecords = dataset_util.open_sharded_output_tfrecords(
                tf_record_close_stack, output_path, num_shards)

            total_num_annotations_skipped = 0
            for idx, annotation_val in enumerate(tqdm(annotations_list)):
                _, tf_example, num_annotations_skipped = self.create_tf_example(annotation_val, verify_dict, feature_spec_dict, idx)
                assert tf_example is not None, 'tf example should not be None!'
                total_num_annotations_skipped += num_annotations_skipped
                shard_idx = idx % num_shards
                output_tfrecords[shard_idx].write(tf_example.SerializeToString())
            tf.logging.info('Finished writing, skipped %d annotations.', total_num_annotations_skipped)

    def evaluate(self, detect_estimator, iou_threshold=0.3, verify_dir=None, limit_count=-1):
        raise NotImplementedError('virtual function')

