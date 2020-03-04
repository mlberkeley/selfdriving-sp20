import modules.globals as globals

import logging
log = logging.getLogger('braking_distance')


import numpy as np
import vision.preset_convs

topdown_mode = False
enable_grayscale = False


cur_conv_n = 0
convs = [None] + vision.preset_convs.convs


def conv_filter(img):
    f = convs[cur_conv_n]
    if f is None:
        return img

    img = img / 255.

    res = [None] * img.shape[2]
    for i in range(img.shape[2]):
        res[i] = vision.preset_convs.conv2d(img[:, :, i], f)
    return np.clip(np.stack(res, axis=2), 0., 1.) * 255.

def render_filter(img):
    if enable_grayscale:
        img = vision.preset_convs.to_grayscale(img)
    img = conv_filter(img)
    if enable_grayscale:
        img = vision.preset_convs.gs_rgb_repr(img)
    return img.astype('uint8')


def next_conv(obs, reward, done, info):
    global cur_conv_n
    cur_conv_n = (cur_conv_n + 1) % len(convs)

def toggle_grayscale(obs, reward, done, info):
    global enable_grayscale
    enable_grayscale = not enable_grayscale

def save_screenshot(obs, reward, done, info):
    obs = render_filter(obs)

    np.save('screen.npy', obs)

    from PIL import Image
    im = Image.fromarray(obs, mode='RGB')
    im.save('screen.png')

def toggle_semantic(obs, reward, done, info):
    globals.env.semantic_mode = not globals.env.semantic_mode

def toggle_mapmode(obs, reward, done, info):
    globals.env.mapmode = not globals.env.mapmode

def toggle_topdown(obs, reward, done, info):
    global topdown_mode
    topdown_mode = not topdown_mode
    globals.render_config['top_down'] = topdown_mode


def register(args):
  from pyglet.window import key

  globals.render_config.update({
      'filter': render_filter,
      'top_down': topdown_mode
  })
  globals.keypress_handlers += [
      (key.N, next_conv),
      (key.G, toggle_grayscale),
      (key.ENTER, save_screenshot),
      (key.P, toggle_semantic),
      (key.M, toggle_mapmode),
      (key.T, toggle_topdown)
  ]

  globals.enable_fns['manual'](args)

  log.info('Vision module enabled')

globals.available_modes += ['vision']
globals.enable_fns['vision'] = register
