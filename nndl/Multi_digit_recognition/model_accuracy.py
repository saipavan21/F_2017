import os
import tensorflow as tf
from data_meta import DataMeta
from data_utils import DataPreprocessor
from cnn_model import CNNModel

tf.app.flags.DEFINE_string('data_dir', './data', 'Directory to read TFRecords files')
tf.app.flags.DEFINE_string('checkpoint_dir', './logs/train', 'Directory to read checkpoint files')
tf.app.flags.DEFINE_string('eval_logdir', './logs/eval', 'Directory to write evaluation logs')
FLAGS = tf.app.flags.FLAGS


class PerformanceCheck(object):

    def __init__(self, path_to_eval_log_dir):
        self.summary_writer = tf.summary.FileWriter(path_to_eval_log_dir)

    def get_accuracy(self, path_to_checkpoint, path_to_tfrecords_file, num_examples, global_step):

    """
        input: path_to_checkpoint => model checkpoint path
               path_to_tfrecords_file => tfrecords path
               num_samples => number of samples to be measured

        funct: evaluates the accuracy of the predicted values of the samples

        output: returns the total accuracy of the model on the sample data given

    """

        batch_size = 128
        num_batches = num_examples / batch_size
        needs_include_length = False

        with tf.Graph().as_default():

            # gets the batches of the evaluting data

            image_batch, length_batch, digits_batch = DataPreprocessor.build_batch(path_to_tfrecords_file,
                                                                         num_examples=num_examples,
                                                                         batch_size=batch_size,
                                                                         shuffled=False)

            length_logits, digits_logits = CNNModel.get_inference(image_batch, drop_rate=0.0)
            length_predictions = tf.argmax(length_logits, axis=1)
            digits_predictions = tf.argmax(digits_logits, axis=2)

            soft = tf.nn.softmax(digits_logits)
            coverage = tf.reduce_max(soft, reduction_indices=2)
            proba = tf.reduce_mean(coverage, axis=1)
            ones = 0.8*tf.ones_like(proba) 
            mask = tf.greater(proba, ones)

            # if length and batch to be concatenated then concatenates

            if needs_include_length:
                labels = tf.concat([tf.reshape(length_batch, [-1, 1]), digits_batch], axis=1)
                predictions = tf.concat([tf.reshape(length_predictions, [-1, 1]), digits_predictions], axis=1)
            else:
                labels = digits_batch
                predictions = digits_predictions

            labels_string = tf.reduce_join(tf.as_string(labels), axis=1)
            predictions_string = tf.reduce_join(tf.as_string(predictions), axis=1)



            labels_mask_string = tf.boolean_mask(labels_string, mask)
            predictions_mask_string = tf.boolean_mask(predictions_string, mask)
            
            coverage_size = tf.size(predictions_mask_string)/tf.size(predictions_string)


            # determining the accuracy of the evaluating data

            accuracy, update_accuracy = tf.metrics.accuracy(
                labels=labels_string,
                predictions=predictions_string
            )

            # determining the accuracy mask of the evaluating data

            accuracy_mask, update_accuracy_mask = tf.metrics.accuracy(
                labels=labels_mask_string,
                predictions=predictions_mask_string
            )

            tf.summary.image('image', image_batch)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.histogram('variables',
                                 tf.concat([tf.reshape(var, [-1]) for var in tf.trainable_variables()], axis=0))
            summary = tf.summary.merge_all()

            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                restorer = tf.train.Saver()
                restorer.restore(sess, path_to_checkpoint)

                for _ in xrange(num_batches):
                    sess.run([update_accuracy, update_accuracy_mask])

                accuracy_val, summary_val = sess.run([accuracy, summary])

                accuracy_mask_val, coverage_size_val = sess.run([accuracy_mask, coverage_size])

                self.summary_writer.add_summary(summary_val, global_step=global_step)

                coord.request_stop()
                coord.join(threads)

        return accuracy_val, accuracy_mask_val, coverage_size_val


def evaluate(checkpoint_dir, path_to_eval_tfrecords_file, num_eval_examples, path_to_eval_log_dir):

"""
    Evaluates the accuracy of the examples and prints them
"""

    performance_check = PerformanceCheck(path_to_eval_log_dir)

    checkpoint_paths = tf.train.get_checkpoint_state(checkpoint_dir).all_model_checkpoint_paths
    for global_step, path_to_checkpoint in [(path.split('-')[-1], path) for path in checkpoint_paths]:
        try:
            global_step_val = int(global_step)
        except ValueError:
            continue

        accuracy, accuracy_mask, coverage_size = performance_check.get_accuracy(path_to_checkpoint, path_to_eval_tfrecords_file,
                                                                                num_eval_examples, global_step_val)
        print('Evaluate %s on %s, accuracy = %f , accuracy_mask = %f, coverage = %f' % (path_to_checkpoint, 
                                            path_to_eval_tfrecords_file, accuracy, accuracy_mask, coverage_size))


def main(_):

"""
    Determines the accuracy of the train, validation, and test data
"""
    # Get the path to train, validation, and test tfrecords files

    train_tfrecords_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    val_tfrecords_file = os.path.join(FLAGS.data_dir, 'val.tfrecords')
    test_tfrecords_file = os.path.join(FLAGS.data_dir, 'test.tfrecords')
    tfrecords_meta_file = os.path.join(FLAGS.data_dir, 'meta.json')
    checkpoint_dir = FLAGS.checkpoint_dir

    # Get the evaluator directory

    train_eval_log_dir = os.path.join(FLAGS.eval_logdir, 'train')
    val_eval_log_dir = os.path.join(FLAGS.eval_logdir, 'val')
    test_eval_log_dir = os.path.join(FLAGS.eval_logdir, 'test')

    # get the meta attributes for the data

    meta = DataMeta()
    meta.load(tfrecords_meta_file)

    # evaluates the accuracy of the examples

    evaluate(checkpoint_dir, train_tfrecords_file, meta.num_train_examples, train_eval_log_dir)
    evaluate(checkpoint_dir, val_tfrecords_file, meta.num_val_examples, val_eval_log_dir)
    evaluate(checkpoint_dir, test_tfrecords_file, meta.num_test_examples, test_eval_log_dir)


if __name__ == '__main__':
    tf.app.run(main=main)
