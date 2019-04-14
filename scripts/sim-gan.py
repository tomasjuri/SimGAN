"""
Implementation of `3.1 Appearance-based Gaze Estimation` from
[Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/pdf/1612.07828v1.pdf).

Note: Only Python 3 support currently.
"""

import os
import sys

from keras import applications
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
import numpy as np
import tensorflow as tf

from config import *
from dlutils import plot_image_batch_w_labels
from utils.image_history_buffer import ImageHistoryBuffer
from data_gen import NegGenerator, PosGenerator, Cropper

from model import discriminator_network, refiner_network

# directories
H5 = '/srv/workplace/tjurica/tasks/1544-ANY_defects_detection/pc_dataset/ann_train.h5'
path = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(path, 'cache')

c = Cropper(H5, dim=(img_width, img_height), crops_per_img=1000)
synthetic_generator = NegGenerator(c, batch_size)
real_generator = PosGenerator(c, batch_size)
    

def adversarial_training(synthetic_generator, real_generator, refiner_model_path=None, discriminator_model_path=None):
    """Adversarial training of refiner network Rθ and discriminator network Dφ."""
    #
    # define model input and output tensors
    #

    synthetic_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    refined_image_tensor = refiner_network(synthetic_image_tensor)

    refined_or_real_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    discriminator_output = discriminator_network(refined_or_real_image_tensor)

    #
    # define models
    #

    refiner_model = models.Model(input=synthetic_image_tensor, output=refined_image_tensor, name='refiner')
    discriminator_model = models.Model(input=refined_or_real_image_tensor, output=discriminator_output,
                                       name='discriminator')

    # combined must output the refined image along w/ the disc's classification of it for the refiner's self-reg loss
    refiner_model_output = refiner_model(synthetic_image_tensor)
    combined_output = discriminator_model(refiner_model_output)
    combined_model = models.Model(input=synthetic_image_tensor, output=[refiner_model_output, combined_output],
                                  name='combined')

    discriminator_model_output_shape = discriminator_model.output_shape

    print(refiner_model.summary())
    print(discriminator_model.summary())
    print(combined_model.summary())

    #
    # define custom l1 loss function for the refiner
    #

    def self_regularization_loss(y_true, y_pred):
        delta = 0.0001  # FIXME: need to figure out an appropriate value for this
        return tf.multiply(delta, tf.reduce_sum(tf.abs(y_pred - y_true)))

    #
    # define custom local adversarial loss (softmax for each image section) for the discriminator
    # the adversarial loss function is the sum of the cross-entropy losses over the local patches
    #

    def local_adversarial_loss(y_true, y_pred):
        # y_true and y_pred have shape (batch_size, # of local patches, 2), but really we just want to average over
        # the local patches and batch size so we can reshape to (batch_size * # of local patches, 2)
        y_true = tf.reshape(y_true, (-1, 2))
        y_pred = tf.reshape(y_pred, (-1, 2))
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        return tf.reduce_mean(loss)

    #
    # compile models
    #

    sgd = optimizers.SGD(lr=0.001)

    refiner_model.compile(optimizer=sgd, loss=self_regularization_loss)
    discriminator_model.compile(optimizer=sgd, loss=local_adversarial_loss)
    discriminator_model.trainable = False
    combined_model.compile(optimizer=sgd, loss=[self_regularization_loss, local_adversarial_loss])


    def get_image_batch(generator):
        """keras generators may generate an incomplete batch for the last batch"""
        img_batch = generator.next()
        if len(img_batch) != batch_size:
            img_batch = generator.next()

        assert len(img_batch) == batch_size

        return img_batch

    # the target labels for the cross-entropy loss layer are 0 for every yj (real) and 1 for every xi (refined)
    y_real = np.array([[[1.0, 0.0]] * discriminator_model_output_shape[1]] * batch_size)
    y_refined = np.array([[[0.0, 1.0]] * discriminator_model_output_shape[1]] * batch_size)
    assert y_real.shape == (batch_size, discriminator_model_output_shape[1], 2)

    if not refiner_model_path:
        # we first train the Rθ network with just self-regularization loss for 1,000 steps
        print('pre-training the refiner network...')
        gen_loss = np.zeros(shape=len(refiner_model.metrics_names))

        for i in range(1000):
            synthetic_image_batch = get_image_batch(synthetic_generator)
            gen_loss = np.add(refiner_model.train_on_batch(synthetic_image_batch, synthetic_image_batch), gen_loss)

            # log every `log_interval` steps
            if not i % log_interval:
                figure_name = 'refined_image_batch_pre_train_step_{}.png'.format(i)
                print('Saving batch of refined images during pre-training at step: {}.'.format(i))

                synthetic_image_batch = get_image_batch(synthetic_generator)
                plot_image_batch_w_labels.plot_batch(
                    np.concatenate((synthetic_image_batch[:,:,:,:3], refiner_model.predict_on_batch(synthetic_image_batch)[:,:,:,:3])),
                    os.path.join(cache_dir, figure_name),
                    label_batch=['Synthetic'] * batch_size + ['Refined'] * batch_size)

                print('Refiner model self regularization loss: {}.'.format(gen_loss / log_interval))
                gen_loss = np.zeros(shape=len(refiner_model.metrics_names))

        refiner_model.save(os.path.join(cache_dir, 'refiner_model_pre_trained.h5'))
    else:
        refiner_model.load_weights(refiner_model_path)

    if not discriminator_model_path:
        # and Dφ for 200 steps (one mini-batch for refined images, another for real)
        print('pre-training the discriminator network...')
        disc_loss = np.zeros(shape=len(discriminator_model.metrics_names))

        for _ in range(100):
            real_image_batch = get_image_batch(real_generator)
            disc_loss = np.add(discriminator_model.train_on_batch(real_image_batch, y_real), disc_loss)

            synthetic_image_batch = get_image_batch(synthetic_generator)
            refined_image_batch = refiner_model.predict_on_batch(synthetic_image_batch)
            disc_loss = np.add(discriminator_model.train_on_batch(refined_image_batch, y_refined), disc_loss)

        discriminator_model.save(os.path.join(cache_dir, 'discriminator_model_pre_trained.h5'))

        # hard-coded for now
        print('Discriminator model loss: {}.'.format(disc_loss / (100 * 2)))
    else:
        discriminator_model.load_weights(discriminator_model_path)

    # TODO: what is an appropriate size for the image history buffer?
    image_history_buffer = ImageHistoryBuffer((0, img_height, img_width, img_channels), batch_size * 100, batch_size)

    combined_loss = np.zeros(shape=len(combined_model.metrics_names))
    disc_loss_real = np.zeros(shape=len(discriminator_model.metrics_names))
    disc_loss_refined = np.zeros(shape=len(discriminator_model.metrics_names))

    # see Algorithm 1 in https://arxiv.org/pdf/1612.07828v1.pdf
    for i in range(nb_steps):
        print('Step: {} of {}.'.format(i, nb_steps))

        # train the refiner
        for _ in range(k_g * 2):
            # sample a mini-batch of synthetic images
            synthetic_image_batch = get_image_batch(synthetic_generator)

            # update θ by taking an SGD step on mini-batch loss LR(θ)
            combined_loss = np.add(combined_model.train_on_batch(synthetic_image_batch,
                                                                 [synthetic_image_batch, y_real]), combined_loss)

        for _ in range(k_d):
            # sample a mini-batch of synthetic and real images
            synthetic_image_batch = get_image_batch(synthetic_generator)
            real_image_batch = get_image_batch(real_generator)

            # refine the synthetic images w/ the current refiner
            refined_image_batch = refiner_model.predict_on_batch(synthetic_image_batch)

            # use a history of refined images
            half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
            image_history_buffer.add_to_image_history_buffer(refined_image_batch)

            if len(half_batch_from_image_history):
                refined_image_batch[:batch_size // 2] = half_batch_from_image_history

            # update φ by taking an SGD step on mini-batch loss LD(φ)
            disc_loss_real = np.add(discriminator_model.train_on_batch(real_image_batch, y_real), disc_loss_real)
            disc_loss_refined = np.add(discriminator_model.train_on_batch(refined_image_batch, y_refined),
                                       disc_loss_refined)

        if not i % log_interval:
            # plot batch of refined images w/ current refiner
            figure_name = 'refined_image_batch_step_{}.png'.format(i)
            print('Saving batch of refined images at adversarial step: {}.'.format(i))

            synthetic_image_batch = get_image_batch(synthetic_generator)
            plot_image_batch_w_labels.plot_batch(
                np.concatenate((synthetic_image_batch[:,:,:,:3], refiner_model.predict_on_batch(synthetic_image_batch)[:,:,:,:3])),
                os.path.join(cache_dir, figure_name),
                label_batch=['Synthetic'] * batch_size + ['Refined'] * batch_size)

            # log loss summary
            print('Refiner model loss: {}.'.format(combined_loss / (log_interval * k_g * 2)))
            print('Discriminator model loss real: {}.'.format(disc_loss_real / (log_interval * k_d * 2)))
            print('Discriminator model loss refined: {}.'.format(disc_loss_refined / (log_interval * k_d * 2)))

            combined_loss = np.zeros(shape=len(combined_model.metrics_names))
            disc_loss_real = np.zeros(shape=len(discriminator_model.metrics_names))
            disc_loss_refined = np.zeros(shape=len(discriminator_model.metrics_names))

            # save model checkpoints
            model_checkpoint_base_name = os.path.join(cache_dir, '{}_model_step_{}.h5')
            refiner_model.save(model_checkpoint_base_name.format('refiner', i))
            discriminator_model.save(model_checkpoint_base_name.format('discriminator', i))


def main():
    adversarial_training(
        synthetic_generator, real_generator,
        refiner_model_path=None, discriminator_model_path=None)

if __name__ == '__main__':
    main()
