import tensorflow as tf


class CNNModel(object):

    @staticmethod
    def get_inference(x, dropout_rate):

        """
            input: x => images tensor
                   dropout_rate => layers dropout rate : dtype => float
            output: length of the logits, and the digits of the output layer 

        """
        # Pool_size = 2*2
        # kernel_size = 5*5

        # Convolution layer 1 with 48 filters with pooling strides =2

        with tf.variable_scope('hidden1'):
            conv = tf.layers.conv2d(x, filters=48, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=dropout_rate)
            hidden1 = dropout

        # Convolution layer 2 with 64 filters with pooling strides =1

        with tf.variable_scope('hidden2'):
            conv = tf.layers.conv2d(hidden1, filters=64, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=dropout_rate)
            hidden2 = dropout

        # Convolution layer 3 with 128 filters with pooling strides =2

        with tf.variable_scope('hidden3'):
            conv = tf.layers.conv2d(hidden2, filters=128, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=dropout_rate)
            conv_layer3 = dropout

        # Convolution layer 4 with 160 filters with pooling strides =1

        with tf.variable_scope('hidden4'):
            conv = tf.layers.conv2d(conv_layer3, filters=160, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=dropout_rate)
            hidden4 = dropout

        # Convolution layer 5 with 192 filters with pooling strides =2

        with tf.variable_scope('hidden5'):
            conv = tf.layers.conv2d(hidden4, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=dropout_rate)
            hidden5 = dropout

        # Convolution layer 6 with 192 filters with pooling strides =1

        with tf.variable_scope('hidden6'):
            conv = tf.layers.conv2d(hidden5, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=dropout_rate)
            hidden6 = dropout

        # Convolution layer 7 with 192 filters with pooling strides =2

        with tf.variable_scope('hidden7'):
            conv = tf.layers.conv2d(hidden6, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=dropout_rate)
            hidden7 = dropout

        # Convolution layer 8 with 192 filters with pooling strides =1

        with tf.variable_scope('hidden8'):
            conv = tf.layers.conv2d(hidden7, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=dropout_rate)
            hidden8 = dropout

        # shallow hidden layer 1 

        flatten = tf.reshape(hidden8, [-1, 4 * 4 * 192])

        # Hidden dense layer 1 with 3072 neurons

        with tf.variable_scope('hidden9'):
            dense = tf.layers.dense(flatten, units=3072, activation=tf.nn.relu)
            hidden9 = dense

        # Hidden dense layer 2 with 3072 neurons

        with tf.variable_scope('hidden10'):
            dense = tf.layers.dense(hidden9, units=3072, activation=tf.nn.relu)
            hidden10 = dense

        # Output layer for length of the digits

        with tf.variable_scope('digit_length'):
            dense = tf.layers.dense(hidden10, units=7)
            length = dense

        # Output layer for digit1

        with tf.variable_scope('digit1'):
            dense = tf.layers.dense(hidden10, units=11)
            digit1 = dense

        # Output layer for digit2

        with tf.variable_scope('digit2'):
            dense = tf.layers.dense(hidden10, units=11)
            digit2 = dense

        # Output layer for digit3

        with tf.variable_scope('digit3'):
            dense = tf.layers.dense(hidden10, units=11)
            digit3 = dense

        # Output layer for digit4

        with tf.variable_scope('digit4'):
            dense = tf.layers.dense(hidden10, units=11)
            digit4 = dense

        # Output layer for digit5

        with tf.variable_scope('digit5'):
            dense = tf.layers.dense(hidden10, units=11)
            digit5 = dense


        length_logits, digits_logits = length, tf.stack([digit1, digit2, digit3, digit4, digit5], axis=1)

        return length_logits, digits_logits

    @staticmethod
    def get_loss(length_logits, digits_logits, length_labels, digits_labels):
        """
            input: length_logits => predicted lengths of the numbers in images
                    digits_logits => predicted digits of the numbers in images
                    length_labels => labeled length of the numbers in images
                    digits_labels => labeled digits of the numbers in images
            
            funct: Calculates the softmax cross entropy loss for length and digits and aggregates the total loss

            output: loss => Total cross entropy loss 

        """
        length_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=length_labels,
                                                                                        logits=length_logits))
        digit1_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 0],
                                                                                        logits=digits_logits[:, 0, :]))
        digit2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 1],
                                                                                        logits=digits_logits[:, 1, :]))
        digit3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 2],
                                                                                        logits=digits_logits[:, 2, :]))
        digit4_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 3],
                                                                                        logits=digits_logits[:, 3, :]))
        digit5_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 4],
                                                                                        logits=digits_logits[:, 4, :]))
        loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy \
                                                            + digit4_cross_entropy + digit5_cross_entropy
        return loss
