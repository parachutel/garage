import gym
import pytest

from garage.experiment import LocalRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import REPS
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
from garage.tf.samplers import OnPolicyVectorizedSampler
from tests.fixtures import TfGraphTestCase

configs = [(1, None, 4), (3, None, 12), (2, 3, 3)]


class TestOnPolicyVectorizedSampler(TfGraphTestCase):
    @pytest.mark.parametrize('cpus, n_envs, expected_n_envs', [*configs])
    def test_on_policy_vectorized_sampler_n_envs(self, cpus, n_envs,
                                                 expected_n_envs):
        """Test REPS with gym Cartpole environment."""
        with LocalRunner(sess=self.sess, max_cpus=cpus) as runner:
            env = TfEnv(gym.make('CartPole-v0'))

            policy = CategoricalMLPPolicy(
                env_spec=env.spec, hidden_sizes=[32, 32])

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = REPS(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                discount=0.99)

            runner.setup(algo, env, sampler_args=dict(n_envs=n_envs))

            assert isinstance(runner.sampler, OnPolicyVectorizedSampler)
            assert runner.sampler.n_envs == expected_n_envs

            env.close()
