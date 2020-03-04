import numpy as np

# Preset convolution kernels

# Identity
kernel_id = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
# 3x3 box blur
kernel_box3 = 1/9 * np.array([
    [1., 1., 1.],
    [1., 1., 1.],
    [1., 1., 1.]
])
# 5x5 box blur
kernel_box5 = 1/25 * np.array([
    [1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1.]
])
# 3x3 gaussian blur
kernel_gauss3 = 1/16 * np.array([
    [1., 2., 1.],
    [2., 4., 2.],
    [1., 2., 1.]
])
# 5x5 gaussian blur
kernel_gauss5 = 1/273 * np.array([
    [1.,  4.,  7.,  4., 1.],
    [4., 16., 26., 16., 4.],
    [7., 26., 41., 26., 7.],
    [4., 16., 26., 16., 4.],
    [1.,  4.,  7.,  4., 1.]
])
# 3x3 sharpen
kernel_sharpen = np.array([
    [ 0., -1.,  0.],
    [-1.,  5., -1.],
    [ 0., -1.,  0.]
])
# 3x3 embossing
kernel_emboss = np.array([
    [ 3.,  2.,  0.],
    [ 2.,  1., -2.],
    [ 0., -2., -3.]
])
# 3x3 left sobel
kernel_lsobel = np.array([
    [ 1.,  0., -1.],
    [ 2.,  0., -2.],
    [ 1.,  0., -1.]
])
# 3x3 left Scharr
kernel_lscharr = np.array([
    [  3.,   0., -3. ],
    [ 10.,   0., -10.],
    [  3.,   0., -3. ]
])
# 3x3 left Prewitt
kernel_lprewitt = np.array([
    [ 1.,  0., -1.],
    [ 1.,  0., -1.],
    [ 1.,  0., -1.]
])
# 3x3 laplace
kernel_laplace = np.array([
    [ 0.,  1., 0.],
    [ 1., -4., 1.],
    [ 0.,  1., 0.]
])
