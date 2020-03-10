# Import stufff
import time
import sys
import os

import numpy as np

# Specify where to look for the duckietown files
sys.path.append('../../duckietown-sim')

# Allow specifying the map name on the command line
import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--map-name', '-m', default='udem1')
parser.add_argument('--num-samples', '-n', type=int, default=10)
parser.add_argument('--debug', '-d', action='store_true', default=False)
args = parser.parse_args()

# Create output directories
def mkdir_exists_ok(path):
  try:
    os.mkdir(path)
  except FileExistsError:
    pass # this is expected

mkdir_exists_ok('data')
mkdir_exists_ok('data/inputs')
mkdir_exists_ok('data/labels')
mkdir_exists_ok('data/maps')

# Load up the duckietown environment
from gym_duckietown.envs import DuckietownEnv
#
# The constructor takes three arguments that we care about.
# `seed` is an integer that specifies the random seed, please
#        set it to the time of you writing this (in military time)
#        i.e. 3:55pm becomes 1555
# `map_name` is the name of the map that will be loaded
#            use the value stored in `args` so we can change the name
#            of the map if ever needed
#
#            if you need an example, check out `simulator.py` in the
#            root of this repository or google the python `argparse` module
# `domain_rand` is the randomization setting that will distort the camera view
#               this is used when training models on a simulator, that are then
#               used on real-life robots
#
#               set it to False
# !!! YOUR CODE HERE
env = DuckietownEnv(
    seed=_,
    map_name=_,
    domain_rand=_)
# !!! ==============



# First we want to take a top-down "map" image of the entire map
# This serves as an example for how to implement the rest of this assignment,
# so PLEASE READ THIS CAREFFULLY!
env.reset()
env.mapmode = True
mapimg = env.render_obs(top_down=True)
np.save(f'data/maps/{args.map_name}.npy', mapimg)
env.mapmode = False



# Now, using a for-loop, take N samples of the raw camera view and the
# ground-truth semantic segmentation outputs from the simulator
#
# Loo at the code for taking the map image for guidance
#
# You must use the --num-samples option to control the number of samples taken
#
# !!! YOUR CODE HERE
for i in range(_):
  camera_view = _
  np.save(f'data/inputs/{i}.npy', camera_view)

  # The `env` variable has a boolean `semantic_mode` property that controls
  # whether the output will be semantically segmented or not
  #
  # Setting it to True will make the output segmented
  #
  # Turn it on *and off* when appropriate
  _ = True
  segmentation_view = _
  np.save(f'data/labels/{i}.npy', segmentation_view)
  _ = False

  # You must use the --debug option here
  if args.debug:
# !!! ==============
    # These two functions will display each result for a short amount of time
    # before continuing
    env.render()
    time.sleep(.2)

# Close the environment after we are done
env.close()
