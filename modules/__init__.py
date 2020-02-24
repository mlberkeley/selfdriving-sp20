import logging
_log = logging.getLogger('modules')

from importlib import import_module
def _try_import(module):
  try:
    # try to import a module, but don't crash if it isn't there
    import_module(f'.{module}', __name__)
  except ModuleNotFoundError as e:
    if not e.name.endswith(module):
      # this exception is something other than just the module missing
      # throw it again
      raise e
    _log.warning(f'Module "{module}" not found, if you need to use this module check if its files exist.')

# modules that we will try to import
_possible_modules = [
  'manual',
  'sysID_data',
  'sysID_test',
  'braking_distance'
]

for m in _possible_modules:
  _try_import(m)

# the util module will always get imported (it auto-enables itself)
# it does things like make ESCAPE quit the app etc.
import modules.globals as globals
import_module('.util', __name__) # don't expose
