import tensorflow as tf
import numpy as np
import sys
sys.path.append('../../duckietown-sim/')
from gym_duckietown.envs import DuckietownEnv

from diffopt import Deterministic

class Dynamics:
    def __init__(self, env):
        self.wheel_dist = tf.constant(env.unwrapped.wheel_dist, dtype=tf.float32)
        self.k = tf.constant(env.k, dtype=tf.float32)
        self.radius = tf.constant(env.radius, dtype=tf.float32)
        self.gain = tf.constant(env.gain, dtype=tf.float32)
        self.trim = tf.constant(env.trim, dtype=tf.float32)
        self.limit = tf.constant(env.limit, dtype=tf.float32)
        self.robot_speed = tf.constant(env.robot_speed, dtype=tf.float32)
        self.delta_time = tf.constant(env.delta_time, dtype=tf.float32)

        self.dynamics_model = Deterministic(lambda time, inputs: self.batch_dynamics(inputs[0], inputs[1]))

    def batch_dynamics(self, states, actions):
        if len(states.shape) < len(actions.shape):
            states = tf.tile(tf.expand_dims(states, 0), tf.constant([actions.shape[0], 1], dtype=tf.int32))

        out = tf.map_fn(lambda inputs: self.dynamics(inputs[0], inputs[1]), (states, actions), dtype=tf.float32)
        return [out]


    def dynamics(self, state, action):
        vel, angle = action[0], action[1]
        wheel_vels = self.get_wheel_vels(vel, angle)

        cur_pos, cur_angle = state[:2], state[2]
        new_pos, new_angle = self.new_pos_angle(cur_pos, cur_angle, wheel_vels)
        out = tf.concat([new_pos, [new_angle]], 0)
        return out

    def get_wheel_vels(self, vel, angle):
        # Distance between the wheels
        baseline = self.wheel_dist

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * baseline) / self.radius
        omega_l = (vel - 0.5 * angle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = tf.clip_by_value(u_r, -self.limit, self.limit)
        u_l_limited = tf.clip_by_value(u_l, -self.limit, self.limit)

        vels = tf.stack([u_l_limited, u_r_limited], 0)
        return vels

    def new_pos_angle(self, cur_pos, cur_angle, vels):
        wheelVels = vels * self.robot_speed * 1
        prev_pos = cur_pos

        # Update the robot's position
        new_pos, new_angle = self._update_pos(cur_pos,
                                        cur_angle,
                                        self.wheel_dist,
                                        wheelVels=wheelVels,
                                        deltaTime=self.delta_time)
        return new_pos, new_angle

    def _update_pos(self, pos, angle, wheel_dist, wheelVels, deltaTime):
        Vl, Vr = wheelVels
        l = wheel_dist

        # If the wheel velocities are the same, then there is no rotation
        if Vl == Vr:
            pos = pos + deltaTime * Vl * self.get_dir_vec(angle)
            return pos, angle

        # Compute the angular rotation velocity about the ICC (center of curvature)
        w = (Vr - Vl) / l

        # Compute the distance to the center of curvature
        r = (l * (Vl + Vr)) / (2 * (Vl - Vr))

        # Compute the rotation angle for this time step
        rotAngle = w * deltaTime

        # Rotate the robot's position around the center of rotation
        r_vec = self.get_right_vec(angle)
        px, pz = pos
        cx = px + r * r_vec[0]
        cz = pz + r * r_vec[1]
        npx, npz = self.rotate_point(px, pz, cx, cz, rotAngle)
        new_pos = tf.stack([npx, npz], 0)

        # Update the robot's direction angle
        new_angle = angle + rotAngle
        return new_pos, new_angle

    def get_dir_vec(self, angle):
        """
        Vector pointing in the direction the agent is looking
        """

        x = tf.cos(angle)
        z = -tf.sin(angle)
        return tf.stack([x, z], 0)

    def get_right_vec(self, angle):
        """
        Vector pointing to the right of the agent
        """

        x = tf.sin(angle)
        z = tf.cos(angle)
        return tf.stack([x, z], 0)

    def rotate_point(self, px, py, cx, cy, theta):
        """
        Rotate a 2D point around a center
        """

        dx = px - cx
        dy = py - cy

        new_dx = dx * tf.cos(theta) + dy * tf.sin(theta)
        new_dy = dy * tf.cos(theta) - dx * tf.sin(theta)

        return cx + new_dx, cy + new_dy

if __name__ == "__main__":
    trajectory = [np.array([2., 1.]), np.array([2., 3.]), np.array([1., 3.]), np.array([1.,1.])]

    env = DuckietownEnv(map_name='udem1', user_tile_start=(1, 1), domain_rand=False,
                        init_x=0.75, init_z=0.75, init_angle=0.)
    env.reset()
    env.render(top_down=True)

    Dyn = Dynamics(env)
    state = [env.cur_pos[0], env.cur_pos[2], env.cur_angle]
    action = [0.1, 0.1]
    pred = Dyn.dynamics(state, action)
    for i in range(10):
        state = [env.cur_pos[0], env.cur_pos[2], env.cur_angle]

        action = [0.1, 0.1]

        print("STATE", state, "PRED", pred)
        tf_action = tf.constant(action, dtype=tf.float32)
        tf_state = tf.constant(state, dtype=tf.float32)
        pred = Dyn.dynamics(tf_state, tf_action)

        env.step(action)
        env.render(top_down=True)

    print("FINISHED")