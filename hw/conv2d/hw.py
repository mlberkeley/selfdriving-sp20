import numpy as np

# Implementation (Grayscale)
def conv2d_grayscale(img, ker):
    """
    Convolve a 2D single-channel image with a 2D single-channel kernel
    using the sliding-window definition.

    img - image (height x width)
    ker - kernel (height x width)
    """

    # Start by decomposing the `.shape`s of the inputs
    # into variables for easier use.
    #
    # See slide 85 for a reference.
    #
    # Note: be very careful with the order of the dimensions!
    #       We used width x height in the lecture, but numpy uses height x width!
    #
    # Here is the numpy documentation of `.shape`:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html
    # !!! YOUR CODE HERE
    j, i = _
    m, n = _
    # !!! ==============

    # Now compute the feature map size. See slide 85 for a reference.
    #
    # You should end up with two numbers (width and height)
    # that are a little smaller than those of the image.
    #
    # !!! YOUR CODE HERE
    Nx = _
    Ny = _
    # !!! ==============

    # Define an empty numpy array of the right size and type for the feature map.
    #
    # There are many numpy functions that could work here, so practice your google-fu
    # to find the most convenient one.
    #
    # You will need to specify the type (i.e. dtype) of the array for it to work properly.
    # Numpy will throw a helpful error if you get it wrong, but make sure you understand
    # why the type is what it is. Hint: think about the format of our image data
    #
    # !!! YOUR CODE HERE
    feature_map = _
    # !!! ==============

    # Now we just need to iterate over the possible kernel locations,
    # and compute the convolution output at each of them.
    #
    #
    # Check slides 76 and later for a computation of the range of the valid x and y values.
    #
    # The iteration part shouldn't do anything fancy at all, just two lines of pure python.
    #
    #
    # For computing the convolution, you'll need to cut out parts of the image.
    # If you are lost on how to do that, check out this numpy documentation page:
    # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    #
    # As a reminder, for convolutions, _each output element_ is defined as
    # a _sum_ of the _element-wise product_ of
    # the kernel and the corresponding part of the image.
    #
    # Google the necessary numpy functions!
    # Our solution only uses one in addition to arithmetic operators.
    #
    #
    # Store the result in the feature_map array.
    #
    # !!! YOUR CODE HERE
    for x in _:
        for y in _:
            feature_map[_, _] = _
    # !!! ==============

    return feature_map

# Implementation (Color)
def conv2d(img, ker):
    """
    Convolve a 2D multi-channel image with a 2D multi-channel kernel
    using the sliding-window definition.

    img - image (height x width x channels)
    ker - kernel (height x width x channels)
    """

    #
    # Feel free to copy parts from your conv2d_grayscale solution!
    #

    # Define the size variables again, this time thinking also
    # about the number of channels that the image and the kernel have.
    #
    # !!! YOUR CODE HERE
    j, i, kc = _
    m, n, ic = _
    # !!! ==============

    # Sanity check!
    #
    # What do we expect the relationship to be between the
    # number of kernel channels and the number of image channels?
    #
    # Use an `assert` statement to verify this.
    #
    # !!! YOUR CODE HERE
    assert kc ???????? ic
    # !!! ==============

    # Define the size of the output feature map again.
    #
    # Did it's size change from the single-channel case?
    #
    # How many channels should the feature map have?
    #
    #
    # !!! YOUR CODE HERE
    Nx = _
    Ny = _
    Nc = _
    # !!! ==============

    # Create an empty array for the feature map again.
    #
    # !!! YOUR CODE HERE
    feature_map = _
    # !!! ==============

    # Iterate over the valid kernel placements and compute the convolution.
    # Again.
    #
    # Don't forget that the image data _and_ the kernel now have
    # _multiple channels_!
    #
    # !!! YOUR CODE HERE
    for x in _:
        for y in _:
            feature_map[_] = _
    # !!! ==============

    return feature_map

# Many Feature Maps
def conv2d_many(img, kers):
    """
    Convolve an image with multiple kernels,
    returning the results as a multi-channel feature map.

    img - image (height x width x channels)
    kers - list of kernels [(height x width x channels)]
    """

    # We already have a convolution function (`conv2d`)!
    #
    # So all we need to do is call it for each kernel.
    # The best way to do this is a list comprehension,
    # but you can use a for loop or any other way of iterating over lists.
    #
    # !!! YOUR CODE HERE
    feature_maps = _
    # !!! ==============

    # Now we need to stack the feature maps in the channel dimension.
    #
    # Numpy provides a function to do that (called, unsurprisingly, `stack`).
    # The only thing you need to figure out is the right value for the `axis` parameter.
    #
    # If you get it wrong, the sanity check will fail,
    # so you can be sure you got it right if it passes.
    #
    # `np.stack` documentation:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.stack.html
    #
    # Hint: look at `stack_kernel` from the Color section.
    #
    # !!! YOUR CODE HERE
    stacked_feature_maps = np.stack(feature_maps, axis=_)
    # !!! ==============

    return stacked_feature_maps

# Padding
def conv2d_padding(img, kers):
    """
    Convolve a 2D multi-channel image with multiple 2D multi-channel kernels,
    padding the image so that the input and the output sizes match.

    img - image (height x width x channels)
    kers - list of kernels [(height x width x channels)]
    """

    # We already have a complete implementation of
    # multi-channel multi-kernel convolutions (conv2d_many).
    #
    # All we need to do is pad the image with zeros before
    # we convolve it!

    # First set the variables that correspond to the dimensions of the kernels
    # Assume all kernels have the same shape, so any one of them (for example the 1st one)
    # can be used to find the shape of the kernels.
    #
    # !!! YOUR CODE HERE
    j, i, kc = _
    # !!! ==============

    # Now compute the padding for each axis of the image
    # (it has three: width, height, channels)
    #
    # Think about how many pixels are missing from
    # the feature map on each axis compared to the image.
    # Since you already have a convolution implementation,
    # you can just look at the `.shape` of any of our examples to check.
    #
    # We need to add enough pixels to the image so that when
    # it shrinks after the convolution it is just the right size.
    # Note that we add the same amount of pixels on each side,
    # so the total number of added pixels is actually
    # on the, for example, x axis is `x_pad * 2`
    #
    # This value should only depend on the size of the kernel.
    #
    # !!! YOUR CODE HERE
    y_pad = _
    x_pad = _
    c_pad = _
    # !!! ==============

    # numpy has a very verbose syntax for padding images
    # it requires that we specify both the before and the after padding
    # for each axis.
    #
    # Since we are padding with the same amount on the top and bottom, and the left and right
    # i.e. top = bottom and left = right
    # we have to just repeat ourselves:
    padding = ((y_pad, y_pad), (x_pad, x_pad), (c_pad, c_pad))

    return conv2d_many(np.pad(img, padding), kers)

# Stride
def conv2d_full(img, kers, stride=1):
    """
    Convolve a 2D multi-channel image with multiple 2D multi-channel kernels,
    padding the image so that the input and the output sizes match and using
    the specified stride.

    img - image (height x width x channels)
    kers - list of kernels [(height x width x channels)]
    stride - stride
    """

    # Take the padding code from `conv2d_padding`.
    # We will reuse it here to pad the image:
    #
    # !!! YOUR CODE HERE
    j, i, kc = _

    y_pad = _
    x_pad = _
    c_pad = _
    # !!! ==============

    padding = ((y_pad, y_pad), (x_pad, x_pad), (c_pad, c_pad))
    img = np.pad(img, padding)

    # Now decompose the shape of the image into variables again, for ease of use.
    # Feel free to copy from `conv2d`!
    #
    # !!! YOUR CODE HERE
    m, n, ic = _
    # !!! ==============

    # We will iterate over the kernels so that we can support multi-channel
    # feature maps like `conv2d_many`
    feature_maps = []
    for ker in kers:
        # Decompose the shape of the current kernel into variables, as usual:
        #
        # !!! YOUR CODE HERE
        j, i, kc = _
        # !!! ==============

        # Now comes the first part that's different with stride not equal to 1.
        #
        # The size of the feature map is going to be a lot smaller if
        # stride is greater than 1.
        #
        # Think about what would happen if instead of computing a feature
        # at every position we did it at every second position?
        # What about every third position?
        # How does the size of the output change?
        #
        # Note: If this is hard to think about, try to implement the next
        #       part of the skeleton code first!
        #
        # !!! YOUR CODE HERE
        Nx = _
        Ny = _
        # !!! ==============

        # Do this one in two steps:
        #
        # 1. Take the `feature_map` and the iteration code straight from
        #    `conv2d_grayscale`.
        #
        # 2. Then make a change to how you compute the slices off the image.
        #    In this new version, x and y should only take on values divisble by
        #    the stride.
        #    E.g. if stride = 2, both x and y should always be even
        #         if stride = 3, both x and y should be one of:
        #                        0, 3, 6, 9, 12, 15, 18, ...
        #
        # !!! YOUR CODE HERE
        feature_map = _
        for x in _:
            for y in _:
                feature_map[_] = _
        # !!! ==============
        feature_maps.append(feature_map)

    # Now take the code from `conv2d_many` that `np.stack`s the feature maps:
    #
    # !!! YOUR CODE HERE
    stacked_feature_maps = np.stack(feature_maps, axis=_)
    # !!! ==============

    return stacked_feature_maps

def maxpool2d(img, size=3, stride=3):
    """
    Compute the 2D maxpool of the specified size of a multi-channel 2D image
    Uses the specified stride.

    img - image (height x width x channels)
    size - the size of the pooling window
    stride - stride
    """

    # We want to reuse the `conv2d_full` code as much as possible,
    # so we just redefine i and j to be our pooling window size.
    i = size
    j = size

    # Do this in steps:
    # 1. Copy all of your code from `conv2d_full`
    # 2. Remove the `i, j, kc = ...` line, we already defined these
    # 3. Cut out the `for ker in kers:` loop.
    #    We just return one `feature_map` since there aren't multiple kernels to deal with
    #
    # In the inner loop:
    # 4. Remove the multiplication by the kernel
    # 5. Replace the summation with `np.max`
    #
    # Keep stride and padding handling exactly the same, they will just work.
    #
    # Below is a skeleton of what your code will look like, but certainly
    # don't re-implement everything, just copy your old code!!!
    #
    # !!! YOUR CODE HERE
    y_pad = _
    x_pad = _
    c_pad = _

    padding = ((y_pad, y_pad), (x_pad, x_pad), (c_pad, c_pad))
    img = np.pad(img, padding)

    m, n, ic = _

    Nx = _
    Ny = _

    feature_map = _
    for x in _:
        for y in _:
            feature_map[_] = _
    # !!! ==============
    return feature_map
