import modules.globals as globals

def reset(obs, reward, done, info):
  globals.env.reset()
  globals.car.reset()
  globals.render()

def kill(obs, reward, done, info):
  import sys

  globals.env.close()
  sys.exit(0)


from pyglet.window import key
globals.keypress_handlers += [
    (key.ESCAPE, kill),
    (key.BACKSPACE, reset)
]
