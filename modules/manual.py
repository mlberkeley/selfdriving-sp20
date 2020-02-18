from importlib import import_module
globals = import_module('..globals', __name__)


import logging
log = logging.getLogger('manual')

def control_hook():
  from pyglet.window import key

  key_handler = globals.key_handler
  car = globals.car

  if key_handler[key.F]:
      car.gear("forward")
  elif key_handler[key.R]:
      car.gear("reverse")

  if key_handler[key.UP]:
      car.pedal("accelerate")
  elif key_handler[key.DOWN]:
      car.pedal("brake")
  else:
      car.pedal("release")

  if key_handler[key.LEFT]:
      car.turn("left")
  elif key_handler[key.RIGHT]:
      car.turn("right")
  else:
      car.turn("release")

def register(args):
    globals.control_hooks += [control_hook]

    log.info('Manual control module enabled')

globals.available_modes += ['manual']
globals.enable_fns['manual'] = register
