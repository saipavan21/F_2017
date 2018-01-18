import tensorflow as tf


class DataPreprocessor(object):
    
    @staticmethod
    def get_preprocessed_image(image):

        """
            input: raw image
            funct: resizes the image to 54*54*3 by random cropping
            output: returns image tensor
        """
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.multiply(tf.subtract(image, 0.5), 2)
        image = tf.reshape(image, [64, 64, 3])
        image = tf.random_crop(image, [54, 54, 3])
        return image


    @staticmethod
    def get_image_attr(filename_queue):

        """
            input: string in a queue runner
            funct: reads and decodes the image
            output: returns image, labeled length, and digits
        """

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'length': tf.FixedLenFeature([], tf.int64),
                'digits': tf.FixedLenFeature([5], tf.int64)
            })

        image = DataPreprocessor.get_preprocessed_image(tf.decode_raw(features['image'], tf.uint8))
        length = tf.cast(features['length'], tf.int32)
        digits = tf.cast(features['digits'], tf.int32)
        return image, length, digits


    @staticmethod
    def create_batch(tfrecords_file_path, num_examples, batch_size, shuffled):

        """
            input: tfrecords_file_path => path to tf records files : dtype => str
                   num_examples => total number of examples : dtype => int
                   batch_size => number of images to get in a batch : dtype => int
                   shuffled => To shuffle the data or not : dtype => Boolean

            funct: gets the batch images from the tf records file 

            output: returns the batch images and their labelled length and digits
        """

        assert tf.gfile.Exists(tfrecords_file_path), '%s not found' % tfrecords_file_path

        filename_queue = tf.train.string_input_producer([tfrecords_file_path], num_epochs=None)
        image, length, digits = DataPreprocessor.get_image_attr(filename_queue)

        min_queue_examples = int(0.4 * num_examples)
        if shuffled:
            image_batch, length_batch, digits_batch = tf.train.shuffle_batch([image, length, digits],
                                                                             batch_size=batch_size,
                                                                             num_threads=2,
                                                                             capacity=min_queue_examples + 3 * batch_size,
                                                                             min_after_dequeue=min_queue_examples)
        else:
            image_batch, length_batch, digits_batch = tf.train.batch([image, length, digits],
                                                                     batch_size=batch_size,
                                                                     num_threads=2,
                                                                     capacity=min_queue_examples + 3 * batch_size)
        return image_batch, length_batch, digits_batch
