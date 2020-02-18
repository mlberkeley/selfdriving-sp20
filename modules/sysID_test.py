import modules.globals as globals

import logging
log = logging.getLogger('sysID_test')

def register(args):
    globals.enable_fns['manual'](args)

    log.info('SysID test module registered')

globals.available_modes += ['sysID_test']
globals.enable_fns['sysID_test'] = register
