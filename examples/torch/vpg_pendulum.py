#!/usr/bin/env python3
"""
This is an example to train a task with VPG algorithm.

Here it runs CartPole-v1 environment with 100 iterations.

Results:
    AverageReturn: 100
    RiseTime: itr 16
"""
import torch

from garage.experiment import run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.samplers import OnPolicyVectorizedSampler
from garage.torch.algos import VPG
from garage.torch.modules import GaussianMLPModule
from garage.torch.policies import GaussianMLPPolicy


def run_task(snapshot_config, *_):
    """Set up environment and algorithm and run the task."""
    with LocalTFRunner() as runner:
        env = TfEnv(env_name='InvertedDoublePendulum-v2')

        policy_module = GaussianMLPModule(
            input_dim=env.spec.observation_space.flat_dim,
            output_dim=env.spec.action_space.flat_dim,
            hidden_sizes=[64, 64],
            hidden_nonlinearity=torch.tanh,
            output_nonlinearity=None)
        policy = GaussianMLPPolicy(env.spec, policy_module)

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = VPG(
            env_spec=env.spec,
            policy=policy,
            optimizer=torch.optim.Adam,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            center_adv=False,
            policy_lr=1e-2)

        runner.setup(algo, env, sampler_cls=OnPolicyVectorizedSampler)
        runner.train(n_epochs=100, batch_size=10000)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
