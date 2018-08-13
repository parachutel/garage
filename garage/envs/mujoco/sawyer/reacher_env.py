import numpy as np

from garage.core.serializable import Serializable
from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnv
from garage.envs.mujoco.sawyer.sawyer_env import Configuration
from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnvWrapper


class ReacherEnv(SawyerEnv):
    def __init__(self,
                 goal_position,
                 start_position=None,
                 randomize_start_position=False,
                 **kwargs):
        def generate_start_goal():
            nonlocal start_position
            if start_position is None or randomize_start_position:
                center = self.sim.data.get_geom_xpos('target2')
                start_position = np.concatenate([center[:2], [0.15]])

            # if randomize_start_position:
            #     offset_x = np.random.uniform(low=-0.3, high=0.3)
            #     offset_y = np.random.uniform(low=-0.2, high=0.2)
            #     offset_z = np.random.uniform(low=0, high=0.3)

            #     start_position[0] += offset_x
            #     start_position[1] += offset_y
            #     start_position[2] += offset_z


            start = Configuration(
                gripper_pos=start_position,
                gripper_state=1,
                object_grasped=False,
                object_pos=[0, 0, -1])
            goal = Configuration(
                gripper_pos=goal_position,
                gripper_state=1,
                object_grasped=False,
                object_pos=[0, 0, -1])

            return start, goal

        def reward_fn(env, achieved_goal, desired_goal, info):
            d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            if env._reward_type == 'sparse':
                return (d < env._distance_threshold).astype(np.float32)

            return 1 - np.exp(d)

        SawyerEnv.__init__(self,
                           start_goal_config=generate_start_goal,
                           reward_fn=reward_fn,
                           **kwargs)

    def get_obs(self):
        gripper_pos = self.gripper_position
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('grip') * dt

        object_pos = self.object_position
        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velp -= grip_velp
        grasped = self.has_object
        if self._control_method == 'task_space_control':
            obs = np.concatenate([gripper_pos])
        elif self._control_method == 'position_control':
            arm_qpos = list()
            arm_qvel = list()
            for i in range(7):
                arm_qpos.append(
                    self.sim.data.get_joint_qpos('right_j{}'.format(i)))
                # Excluding the joint velocities since mujoco has different dynamics from the real robot.
                # arm_qvel.append(self.sim.data.get_joint_qvel('right_j{}'.format(i)))

            obs = np.concatenate([arm_qpos, gripper_pos])
        else:
            raise NotImplementedError

        achieved_goal = self._achieved_goal_fn(self)
        desired_goal = self._desired_goal_fn(self)

        achieved_goal_qpos = np.concatenate((achieved_goal, [1, 0, 0, 0]))
        self.sim.data.set_joint_qpos('achieved_goal:joint', achieved_goal_qpos)
        desired_goal_qpos = np.concatenate((desired_goal, [1, 0, 0, 0]))
        self.sim.data.set_joint_qpos('desired_goal:joint', desired_goal_qpos)

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal,
            'gripper_state': self.gripper_state,
            'gripper_pos': gripper_pos.copy(),
            'has_object': grasped,
            'object_pos': object_pos.copy()
        }


class SimpleReacherEnv(SawyerEnvWrapper, Serializable):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.reward_range = None
        self.metadata = None
        super().__init__(ReacherEnv(*args, **kwargs))
