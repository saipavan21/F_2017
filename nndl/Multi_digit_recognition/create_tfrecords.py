import os
import numpy as np
import h5py
import random
from PIL import Image
import tensorflow as tf
from data_meta import DataMeta

tf.app.flags.DEFINE_string('data_dir', './data',
                           'Directory to SVHN (format 1) folders and write the converted files')
FLAGS = tf.app.flags.FLAGS


class SampleReader(object):
    def __init__(self, path_to_image_files):
        self._path_to_image_files = path_to_image_files
        self._num_examples = len(self._path_to_image_files)
        self._sample_pointer = 0

    @staticmethod
    def get_attrs(digit_struct_mat_file, index):

        """
            Returns a dictionary which contains keys: label, left, top, width and height, each key has multiple values.
        """
        attrs = {}
        f = digit_struct_mat_file
        item = f['digitStruct']['bbox'][index].item()
        for key in ['label', 'left', 'top', 'width', 'height']:
            attr = f[item][key]
            values = [f[attr.value[i].item()].value[0][0]
                      for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
            attrs[key] = values
        return attrs

    @staticmethod
    def do_preprocess(image, bbox_left, bbox_top, bbox_width, bbox_height):

        """
            Returns the preprocessed image by cropping and resizing
        """
        cropped_left, cropped_top, cropped_width, cropped_height = (int(round(bbox_left - 0.15 * bbox_width)),
                                                                    int(round(bbox_top - 0.15 * bbox_height)),
                                                                    int(round(bbox_width * 1.3)),
                                                                    int(round(bbox_height * 1.3)))
        image = image.crop([cropped_left, cropped_top, cropped_left + cropped_width, cropped_top + cropped_height])
        image = image.resize([64, 64])
        return image

    @staticmethod
    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def read_and_convert(self, digit_struct_mat_file):

        """
            Read and convert to sample, returns None if no data is available.
        """
        if self._sample_pointer == self._num_examples:
            return None
        path_to_image_file = self._path_to_image_files[self._sample_pointer]
        index = int(path_to_image_file.split('/')[-1].split('.')[0]) - 1
        self._sample_pointer += 1

        # get sample attributes

        attrs = SampleReader.get_attrs(digit_struct_mat_file, index)
        label_of_digits = attrs['label']
        length = len(label_of_digits)

        # if length is greater than 5 skip this example

        if length > 5:
            return self.read_and_convert(digit_struct_mat_file)

        digits = [10, 10, 10, 10, 10]   # digit 10 represents no digit
        for idx, label_of_digit in enumerate(label_of_digits):
            digits[idx] = int(label_of_digit if label_of_digit != 10 else 0)    # label 10 is essentially digit zero

        # gets the bounding boxes of the digits to crop the image

        attrs_left, attrs_top, attrs_width, attrs_height = map(lambda x: [int(i) for i in x], 
                                                                        [attrs['left'], attrs['top'], attrs['width'], attrs['height']])
        min_left, min_top, max_right, max_bottom = (min(attrs_left),
                                                    min(attrs_top),
                                                    max(map(lambda x, y: x + y, attrs_left, attrs_width)),
                                                    max(map(lambda x, y: x + y, attrs_top, attrs_height)))
        center_x, center_y, max_side = ((min_left + max_right) / 2.0,
                                        (min_top + max_bottom) / 2.0,
                                        max(max_right - min_left, max_bottom - min_top))
        bbox_left, bbox_top, bbox_width, bbox_height = (center_x - max_side / 2.0,
                                                        center_y - max_side / 2.0,
                                                        max_side,
                                                        max_side)
        image = np.array(SampleReader.do_preprocess(Image.open(path_to_image_file), bbox_left, bbox_top, bbox_width, bbox_height)).tobytes()

        sample = tf.train.sample(features=tf.train.Features(feature={
            'image': SampleReader.bytes_feature(image),
            'length': SampleReader.int64_feature(length),
            'digits': tf.train.Feature(int64_list=tf.train.Int64List(value=digits))
        }))
        return sample


def create_tfrecords_meta_file(num_train_examples, num_val_examples, num_test_examples,
                               tfrecords_meta_file):

    """
        Saves the meta file to the tfrecords meta file
    """

    meta = DataMeta()
    meta.num_train_examples = num_train_examples
    meta.num_val_examples = num_val_examples
    meta.num_test_examples = num_test_examples
    meta.save(tfrecords_meta_file)


def convert_to_tfrecords(path_to_dataset_dir_and_digit_struct_mat_file_tuples,
                         path_to_tfrecords_files, choose_writer_callback):

    """
        input: path_to_dataset_dir_and_digit_struct_mat_file_tuples => (datset dir,digit_struct_mat_file)
               path_to_tfrecords_files => tfrecords file path

        funct: converts the image and writes them in tfrecords file

        output: returns the number of examples converted

    """
    num_examples = []
    writers = []

    for path_to_tfrecords_file in path_to_tfrecords_files:
        num_examples.append(0)
        writers.append(tf.python_io.TFRecordWriter(path_to_tfrecords_file))

    for path_to_dataset_dir, path_to_digit_struct_mat_file in path_to_dataset_dir_and_digit_struct_mat_file_tuples:
        path_to_image_files = tf.gfile.Glob(os.path.join(path_to_dataset_dir, '*.png'))
        total_files = len(path_to_image_files)


        with h5py.File(path_to_digit_struct_mat_file, 'r') as digit_struct_mat_file:
            sample_reader = SampleReader(path_to_image_files)
            for index, path_to_image_file in enumerate(path_to_image_files):

                sample = sample_reader.read_and_convert(digit_struct_mat_file)
                if sample is None:
                    break

                idx = choose_writer_callback(path_to_tfrecords_files)
                writers[idx].write(sample.SerializeToString())
                num_examples[idx] += 1

    for writer in writers:
        writer.close()

    return num_examples




def main(_):

    """
        Process the train, validation , and test data and create the tfrecords file for each set
    """

    # Getting the data and their meta files path
    train_dir = os.path.join(FLAGS.data_dir, 'train')
    test_dir = os.path.join(FLAGS.data_dir, 'test')
    extra_dir = os.path.join(FLAGS.data_dir, 'extra')
    train_digit_struct_mat_file = os.path.join(train_dir, 'digitStruct.mat')
    test_digit_struct_mat_file = os.path.join(test_dir, 'digitStruct.mat')
    extra_digit_struct_mat_file = os.path.join(extra_dir, 'digitStruct.mat')

    train_tfrecords_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    val_tfrecords_file = os.path.join(FLAGS.data_dir, 'val.tfrecords')
    test_tfrecords_file = os.path.join(FLAGS.data_dir, 'test.tfrecords')
    tfrecords_meta_file = os.path.join(FLAGS.data_dir, 'meta.json')



    for path_to_file in [train_tfrecords_file, val_tfrecords_file, test_tfrecords_file]:
        assert not os.path.exists(path_to_file), 'The file %s already exists' % path_to_file

    # Processing the train and Validation data

    [num_train_examples, num_val_examples] = convert_to_tfrecords([(train_dir, train_digit_struct_mat_file),
                                                                   (extra_dir, extra_digit_struct_mat_file)],
                                                                  [train_tfrecords_file, val_tfrecords_file],
                                                                  lambda paths: 0 if random.random() > 0.1 else 1)
    
    # Processing test data

    [num_test_examples] = convert_to_tfrecords([(test_dir, test_digit_struct_mat_file)],
                                               [test_tfrecords_file],
                                               lambda paths: 0)

    # Create tfrecords fmeta file for train , validation and test data

    create_tfrecords_meta_file(num_train_examples, num_val_examples, num_test_examples,
                               tfrecords_meta_file)

    print ('Completed processing')


if __name__ == '__main__':
    tf.app.run(main=main)
