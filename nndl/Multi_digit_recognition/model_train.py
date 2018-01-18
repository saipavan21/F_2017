import os
from datetime import datetime
import time
import tensorflow as tf
from data_meta import DataMeta
from data_utils import DataPreprocessor
from cnn_model import CNNModel
from model_accuracy import PerformanceCheck

tf.app.flags.DEFINE_string('data_dir', './data', 'Directory to read TFRecords files')
tf.app.flags.DEFINE_string('train_logdir', './logs/train', 'Directory to write training logs')
tf.app.flags.DEFINE_string('restore_checkpoint', None,
                           'Path to restore checkpoint (without postfix), e.g. ./logs/train/model.ckpt-100')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Default 32')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Default 1e-2')
tf.app.flags.DEFINE_integer('patience', 100, 'Default 100, set -1 to train infinitely')
tf.app.flags.DEFINE_integer('decay_steps', 10000, 'Default 10000')
tf.app.flags.DEFINE_float('decay_rate', 0.9, 'Default 0.9')
FLAGS = tf.app.flags.FLAGS


def _train(train_tfrecords_path, num_train_examples, val_tfrecords_path, num_val_examples,
           train_log_dir_path, checkpoint_file_path, training_params):

    """
        input:  train_tfrecords_path => path to train tfrecords file : dtype => str
                num_train_examples => number of training examples : dtype => int
                val_tfrecords_path => path to validation tfrecords file : dtype => str
                num_val_examples => number of validation examples : dtype => int
                train_log_dir_path => path train log directory : dtype => str
                checkpoint_file_path => path to checkpoint file : dtype => str
                training_params => batch_size, patience, learning_rate, decay_rate, decay_steps : dtype => dict

        funct:  Trains the model with the training_params in batches and saves the model in train_log_dir_path
                if the validation accuracy increases, and breaks if the patience level reaches to zero

    """
    batch_size = training_params['batch_size']
    initial_patience = training_params['patience']
    num_steps_to_show_loss = 100
    num_steps_to_check = 1000

    with tf.Graph().as_default():

        #Getting Batch images

        batch_images, batch_length, batch_digits = DataPreprocessor.create_batch(train_tfrecords_path,
                                                                     num_examples=num_train_examples,
                                                                     batch_size=batch_size,
                                                                     shuffled=True)

        # getting the predicted length and digits and finding the cross entropy loss

        logtis_length, logits_digits = CNNModel.get_inference(batch_images, dropout_rate=0.2)
        loss = CNNModel.get_loss(logtis_length, logits_digits, batch_length, batch_digits)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        #Adjusting the learning rate and training using gradient optimizer

        learning_rate = tf.train.exponential_decay(training_params['learning_rate'], global_step=global_step,
                                                   decay_steps=training_params['decay_steps'],
                                                   decay_rate=training_params['decay_rate'], 
                                                   staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_optimizer = optimizer.minimize(loss, global_step=global_step)

        # Creates the summary of the batch images

        tf.summary.image('image', batch_images)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        summary = tf.summary.merge_all()

        #Starting tf session

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(train_log_dir_path, sess.graph)
            performance_check = PerformanceCheck(os.path.join(train_log_dir_path, 'eval/val'))

            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            saver = tf.train.Saver()

            # Fetch the latest model from checkpoint if exists

            if checkpoint_file_path is not None:
                assert tf.train.checkpoint_exists(checkpoint_file_path), \
                    '%s not found' % checkpoint_file_path
                saver.restore(sess, checkpoint_file_path)
                print ('Model restored from file: %s' % checkpoint_file_path)

            print ('-------------Started training--------------')
            patience = initial_patience
            best_accuracy = 0.0
            total_time_taken = 0.0


            while True:
                start_time = time.time()
                _, loss_val, summary_value, global_step_value, learning_rate_value = sess.run([train_optimizer, loss, summary, 
                                                                                        global_step, learning_rate])
                total_time_taken += time.time() - start_time

                if global_step_value % num_steps_to_show_loss == 0:
                    examples_per_sec = batch_size * num_steps_to_show_loss / total_time_taken
                    total_time_taken = 0.0
                    print ('=> %s: step %d, loss = %f (%.1f examples/sec)' % (
                        datetime.now(), global_step_value, loss_val, examples_per_sec))

                if global_step_value % num_steps_to_check != 0:
                    continue

                summary_writer.add_summary(summary_value, global_step=global_step_value)

                # Caluculates the validation accuracy and coverage 

                print ('=> Evaluating on validation dataset...')
                latest_checkpoint_file_path = saver.save(sess, os.path.join(train_log_dir_path, 'latest.ckpt'))
                accuracy, accuracy_mask, coverage = performance_check.get_accuracy(latest_checkpoint_file_path, val_tfrecords_path,
                                              num_val_examples, global_step_value)

                print ('==> accuracy = %f, best accuracy %f' % (accuracy, best_accuracy))
                print('==> accuracy_mask = %f, coverage = %f' % (accuracy_mask, coverage))


                # Saves the best model if accuracy increases

                if accuracy > best_accuracy:
                    path_to_checkpoint_file = saver.save(sess, os.path.join(train_log_dir_path, 'model.ckpt'),
                                                         global_step=global_step_value)
                    print ('=> Model saved to file: %s' % path_to_checkpoint_file)
                    patience = initial_patience
                    best_accuracy = accuracy
                else:
                    patience -= 1

                # Stops training if the patience level reaches zero

                print ('=> patience = %d' % patience)
                if patience == 0:
                    break

            coord.request_stop()
            coord.join(threads)
            print ('Finished')


def main(_):

    """
        funct: gets tfrecords file paths and training training_params and trains the model
    """

    # Getting tfrecords path for train and validation data

    train_tfrecords_path = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    val_tfrecords_path = os.path.join(FLAGS.data_dir, 'val.tfrecords')
    tfrecords_meta_file_path = os.path.join(FLAGS.data_dir, 'meta.json')
    train_log_dir_path = FLAGS.train_logdir
    checkpoint_file_path = FLAGS.restore_checkpoint

    # Getting the number of train and validation examples

    meta = DataMeta()
    meta.load(tfrecords_meta_file_path)

    # Getting training parameters

    training_params = {
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate,
        'patience': FLAGS.patience,
        'decay_steps': FLAGS.decay_steps,
        'decay_rate': FLAGS.decay_rate
    }


    # Training the model

    _train(train_tfrecords_path, meta.num_train_examples,
           val_tfrecords_path, meta.num_val_examples,
           train_log_dir_path, checkpoint_file_path,
           training_params)


if __name__ == '__main__':
    tf.app.run(main=main)
