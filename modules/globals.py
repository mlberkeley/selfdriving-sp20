# populated by every single module with its name to allow enabling it later
available_modes = []
# functions that will add custom arguments to an argparse parser as needed
arg_setup_fns = {}
# functions that will register all hooks, handlers, and configs required by
# modules
# simulator will call one of these, depending on --mode
# modules may call other modules' enable_fn if they want to include that
# functionality as well (e.g. run enable_fns['manual'] to enable manual
# driving in your module)
enable_fns = {}

# one-shot key event handlers
# these will only fire once per keypress, instead of every frame
# use control_hooks for buttons that need to be held down, and
# this for buttons that toggle things/trigger events
keypress_handlers = []
# parsed as **kwargs to env.render, use to influence the rendering result
# in a way that is compatible with multiple modules
render_config = {}
# ran every tick, before env.step, use this for controlling movement, etc.
# ALSO: see keypress_handlers
control_hooks = []

# globals shared from simulator
env = None
car = None
key_handler = None
render = None

# do absolutely nothing no matter what
def noop(*args, **kwargs):
  pass

def setup_arguments_for(name, parser):
    # most modules do NOT setup custom arguments,
    # so we do nothing (noop) for them by default
    arg_setup_fns.get(name, noop)(parser)

