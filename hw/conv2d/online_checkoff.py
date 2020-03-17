import numpy as np

from hw import maxpool2d, conv2d_full
from utils import imload, imsave_grayscale

kodim = imload('kodim20.png')

k1 = np.array([
    [ -1.,  1.,  3.],
    [  1.,  5., -1.],
    [ -3., -6.,  1.]
])
k2 = np.array([
    [ -1.,   4.,   3.],
    [-17.,  17., -10.],
    [ -3.,  -4.,   1.]
])
k3 = np.array([
    [ 7., 19.,   0.],
    [-7.,  0., -19.],
    [ 0.,  3.,  -3.]
])

stack1 = np.stack([k1, k1, k3], axis=2)
stack2 = np.stack([k3, k1, k3], axis=2)
stack3 = np.stack([k3, k2, k2], axis=2)

test = conv2d_full(kodim, [stack1, stack2, stack3], stride=3)
test = maxpool2d(test, size=7, stride=5)

imsave_grayscale(test, 'online_checkoff.png')
