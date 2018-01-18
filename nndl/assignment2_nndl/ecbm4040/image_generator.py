#!/usr/bin/env/ python
# ECBM E4040 Fall 2017 Assignment 2
# This Python script contains the ImageGenrator class.

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate


class ImageGenerator(object):

    def __init__(self, x, y):
        """
        Initialize an ImageGenerator instance.
        :param x: A Numpy array of input data. It has shape (num_of_samples, height, width, channels).
        :param y: A Numpy vector of labels. It has shape (num_of_samples, ).
        """

        # TODO: Your ImageGenerator instance has to store the following information:
        # x, y, num_of_samples, height, width, number of pixels translated, degree of rotation, is_horizontal_flip,
        # is_vertical_flip, is_add_noise. By default, set boolean values to
        # False.
        #raise NotImplementedError
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        self.x =x
        self.y = y
        self.num_of_samples = x.shape[0]
        self.height = x.shape[1]
        self.width = x.shape[2]
        self.channels = x.shape[3]
        self.num_pix_trans = {"height":0, "width":0}
        self.deg_rot = None 
        self.is_horizontal_flip, self.is_vertical_flip, self.is_add_noise = False, False, False


    def next_batch_gen(self, batch_size, shuffle=True):
        """
        A python generator function that yields a batch of data indefinitely.
        :param batch_size: The number of samples to return for each batch.
        :param shuffle: If True, shuffle the entire dataset after every sample has been returned once.
                        If False, the order or data samples stays the same.
        :return: A batch of data with size (batch_size, width, height, channels).
        """

        # TODO: Use 'yield' keyword, implement this generator. Pay attention to the following:
        # 1. The generator should return batches endlessly.
        # 2. Make sure the shuffle only happens after each sample has been visited once. Otherwise some samples might
        # not be output.

        # One possible pseudo code for your reference:
        #######################################################################
        #   calculate the total number of batches possible (if the rest is not sufficient to make up a batch, ignore)
        #   while True:
        #       if (batch_count < total number of batches possible):
        #           batch_count = batch_count + 1
        #           yield(next batch of x and y indicated by batch_count)
        #       else:
        #           shuffle(x)
        #           reset batch_count
        #raise NotImplementedError
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        num_of_samples = self.num_of_samples
        total_batches = int(num_of_samples/batch_size)
        batch_count = 0
        x = self.x
        y = self.y
        while True:
            if batch_count < total_batches:
                batch_count += 1
                start_index = (batch_count -1) * batch_size
                end_index = start_index + batch_size
                yield x[start_index:end_index], y[start_index:end_index]
            else:
                x, y = np.random.shuffle(self.x, self.y)
                batch_count = 0



    def show(self):
        """
        Plot the top 16 images (index 0~15) of self.x for visualization.
        """
        #raise NotImplementedError
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        x_batch = self.x[:16]
        r = 4
        f, axarr = plt.subplots(r, r, figsize=(8,8))
        for i in range(r):
            for j in range(r):
                img = x_batch[r*i+j]
                axarr[i][j].imshow(img)

    def translate(self, shift_height, shift_width):
        """
        Translate self.x by the values given in shift.
        :param shift_height: the number of pixels to shift along height direction. Can be negative.
        :param shift_width: the number of pixels to shift along width direction. Can be negative.
        :return:
        """

        # TODO: Implement the translate function. Remember to record the value of the number of pixels translated.
        # Note: You may wonder what values to append to the edge after the translation. Here, use rolling instead. For
        # example, if you translate 3 pixels to the left, append the left-most 3 columns that are out of boundary to the
        # right edge of the picture.
        # Hint: Numpy.roll
        # (https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.roll.html)
        #raise NotImplementedError
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        shift_array = [0, shift_height, shift_width, 0]
        self. num_pix_trans["height"] += shift_height
        self. num_pix_trans["width"] += shift_width
        x = self.x
        x_trans = np.roll(np.roll(np.roll(np.roll(x, shift_array[0], axis=0), shift_array[1], axis=1), shift_array[2], axis=2), shift_array[3], axis=3)
        self.x = x_trans


    def rotate(self, angle=0.0):
        """
        Rotate self.x by the angles (in degree) given.
        :param angle: Rotation angle in degrees.

        - https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.interpolation.rotate.html
        """
        # TODO: Implement the rotate function. Remember to record the value of
        # rotation degree.
        #raise NotImplementedError
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        self.deg_rot = angle
        x = self.x
        x = rotate(x, angle, axes=(1,2))
        self.x = x

    def flip(self, mode='h'):
        """
        Flip self.x according to the mode specified
        :param mode: 'h' or 'v' or 'hv'. 'h' means horizontal and 'v' means vertical.
        """
        # TODO: Implement the flip function. Remember to record the boolean values is_horizontal_flip and
        # is_vertical_flip.
        #raise NotImplementedError
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        x = self.x
        if mode == 'h':
            self.is_horizontal_flip = True
            x = np.flip(x, axis=2)
        elif mode == 'v':
            self.is_vertical_flip = True
            x = np.flip(x, axis=1)
        else:
            self.is_vertical_flip = True
            self.is_horizontal_flip = True
            x = np.flip(x, axis=1)
            x = np.flip(x, axis=2)

        self.x = x

    def add_noise(self, portion, amplitude):
        """
        Add random integer noise to self.x.
        :param portion: The portion of self.x samples to inject noise. If x contains 10000 sample and portion = 0.1,
                        then 1000 samples will be noise-injected.
        :param amplitude: An integer scaling factor of the noise.
        """
        # TODO: Implement the add_noise function. Remember to record the
        # boolean value is_add_noise. You can try uniform noise or Gaussian
        # noise or others ones that you think appropriate.
        #raise NotImplementedError
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        self.is_add_noise = True
        x = self.x.astype(float)
        noise_sample = int(x.shape[0] * portion)
        rand_ind = np.random.choice(x.shape[0], size=noise_sample, replace=False)
        mean = 0
        std = 0.1**0.5
        #uni_noise = np.random.uniform(0,1, (noise_sample, self.height, self.width, self.channels))
        #uni_noise = uni_noise * amplitude
        #x[rand_ind] = x[rand_ind] + uni_noise
        gauss_noise = np.random.normal(mean, std, (noise_sample, self.height, self.width, self.channels))
        gauss_noise = gauss_noise * amplitude
        x[rand_ind] = x[rand_ind] + gauss_noise
        self.x = x
