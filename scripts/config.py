
#
# image dimensions
#

img_width = 64
img_height = 64
img_channels = 4

#
# training params
#

nb_steps = 10000
batch_size = 256
k_d = 1  # number of discriminator updates per step
k_g = 2  # number of generative network updates per step
log_interval = 100