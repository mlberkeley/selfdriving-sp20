import modules.globals as globals

import logging
log = logging.getLogger('sysID_data')


from pyglet.window import key
def control_hook():
    key_handler = globals.key_handler
    car = globals.car

    if key_handler[key.S]:
        car.save_data()

def register(args):
    globals.control_hooks += [control_hook]

    globals.enable_fns['manual'](args)

    log.info('SysID data module registered')

globals.available_modes += ['sysID_data']
globals.enable_fns['sysID_data'] = register
