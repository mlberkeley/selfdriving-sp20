#!/usr/bin/env python

import logging
log = logging.getLogger('simulator')

# parse once, ignore --help, ignore unknown arguments
# all we care about is --mode
import argparse
parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
parser.add_argument('--mode', default='manual')
args = parser.parse_known_args()[0]


# we have to setup settings + sys.path before importing any modules
# since modules may import files that depend on internal components
import settings
import sys
if settings.use_internal_if_available:
    sys.path.append('../internal_only')


# import all available modules
import modules

# check the mode
if not args.mode in modules.globals.available_modes:
    modes_str = ', '.join([f'"{m}"' for m in modules.globals.available_modes])

    log.error(f'Mode "{args.mode}" not found!')
    log.error(f'Available modes are: {modes_str}')
    sys.exit(1)

# create a new argument parser, this time we will properly parse all the
# arguments, since we already know what module we want to enable and so
# can ask it to register all custom arguments
parser = argparse.ArgumentParser(parents=[parser])
# this feature is disabled for now since we need shared arguments
# but we can't add duplicate arguments
# modules.globals.setup_arguments_for(args.mode, parser)

parser.add_argument('--car_model', default='simple')
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()


modules.globals.enable_fns[args.mode](args)


sys.path.append('duckietown-sim')
from gym_duckietown.envs import DuckietownEnv
env = DuckietownEnv(
    seed = args.seed,
    map_name = args.map_name,
    domain_rand=False)
modules.globals.env = env

def render():
    env.render(**modules.globals.render_config)
render()
modules.globals.render = render


from importlib import import_module
from car_iface.Car import Car
car = Car(args.mode, args.car_model)
car.start()
modules.globals.car = car


from pyglet.window import key
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)
modules.globals.key_handler = key_handler

keys_down = {}

def update(dt):
    for h in modules.globals.control_hooks:
        h()

    import numpy as np
    action = np.array(car.duckietown_control())
    if key_handler[key.LSHIFT] and key_handler[key.UP]:
        action *= 1.5 # do we need this?

    obs, reward, done, info = env.step(action)
    for (k, f) in modules.globals.keypress_handlers:
        global keys_down
        if key_handler[k]:
            if k not in keys_down:
                keys_down[k] = False
            if not keys_down[k]:
                f(obs, reward, done, info)
                keys_down[k] = True
        else:
            keys_down[k] = False

    if done:
        print('\nCrashed!')
        env.reset()
        car.reset()
        render()


import time
update_delay = 1.0 / env.unwrapped.frame_rate
next_update = time.time()
def tick(dt):
    t = time.time()

    global next_update
    if t >= next_update:
        update(dt)
        next_update = t + update_delay

    render()

import pyglet
pyglet.clock.schedule_interval(tick, 1.0 / 60)
pyglet.app.run()


env.close()
