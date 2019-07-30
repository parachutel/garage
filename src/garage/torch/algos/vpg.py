"""Vanilla Policy Gradient."""
import collections

from dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F  # noqa

from garage.misc import special
from garage.np.algos import RLAlgorithm
from garage.tf.misc import tensor_utils


class VPG(RLAlgorithm):
    """Vanilla Policy Gradient.

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.torch.policies.base.Policy): Policy.
        baseline : The baseline.
        scope (str): Scope for identifying the algorithm.
            Must be specified if running multiple algorithms
            simultaneously, each using different environments
            and policies.
        max_path_length (int): Maximum length of a single rollout.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        fixed_horizon (bool): Whether to fix horizon.
        optimizer (object): The optimizer of the algorithm. Should be the
            optimizers in torch.optim.
        optimizer_args (dict): The arguments of the optimizer.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        use_neg_logli_entropy (bool): Whether to estimate the entropy as the
            negative log likelihood of the action.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.
    """

    def __init__(
            self,
            env_spec,
            policy,
            baseline,
            max_path_length=500,
            policy_lr=1e-2,
            discount=0.99,
            gae_lambda=1,
            center_adv=True,
            positive_adv=False,
            max_kl_step=0.01,
            optimizer=None,
            policy_ent_coeff=0.0,
            use_softplus_entropy=False,
            stop_entropy_gradient=False,
            entropy_method='no_entropy',
    ):
        self.env_spec = env_spec
        self.policy = policy
        self.baseline = baseline
        self.max_path_length = max_path_length
        self.policy_lr = policy_lr
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.max_kl_step = max_kl_step
        self.policy_ent_coeff = policy_ent_coeff
        self.use_softplus_entropy = use_softplus_entropy
        self.stop_entropy_gradient = stop_entropy_gradient
        self.entropy_method = entropy_method
        self.eps = 1e-8

        self.maximum_entropy = (entropy_method == 'max')
        self.entropy_regularzied = (entropy_method == 'regularized')
        #     Entropy regularization, max entropy rl

        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          policy_ent_coeff)

        self.reward_mean = collections.deque(maxlen=100)

        self.optimizer = optimizer(
            self.policy.get_module().parameters(), lr=self.policy_lr)

    def _check_entropy_configuration(self, entropy_method, center_adv,
                                     stop_entropy_gradient, policy_ent_coeff):
        if entropy_method not in ('max', 'regularized', 'no_entropy'):
            raise ValueError('Invalid entropy_method')

        if entropy_method == 'max':
            if center_adv:
                raise ValueError('center_adv should be False when '
                                 'entropy_method is max')
            if not stop_entropy_gradient:
                raise ValueError('stop_gradient should be True when '
                                 'entropy_method is max')
        if entropy_method == 'no_entropy':
            if policy_ent_coeff != 0.0:
                raise ValueError('policy_ent_coeff should be zero '
                                 'when there is no entropy method')

    def train_once(self, itr, paths):
        """Perform one step of policy optimization."""
        self._optimize_policy(itr, paths)
        self._optimize_baseline(itr, paths)

        return np.mean([sum(path['rewards']) for path in paths])

    def _optimize_policy(self, itr, paths):
        loss = [self._get_policy_loss(path) for path in paths]

        self.optimizer.zero_grad()
        torch.cat(loss).mean().backward()
        self.optimizer.step()

    def _get_policy_loss(self, path):
        rewards = torch.Tensor(path['rewards'])
        obs = torch.Tensor(path['observations'])

        policy_entropy = self._get_policy_entropy(obs)

        if self.maximum_entropy:
            rewards = [
                reward + self.policy_ent_coeff * policy_entropy
                for reward in rewards
            ]

        baselines = self._get_baselines(path)
        path['baselines'] = baselines.detach().numpy()
        advantages = self._compute_advantages(baselines, rewards)

        if self.center_adv:
            advantages = F.batch_norm(
                advantages.reshape((-1, 1)),
                advantages.mean().unsqueeze(0),
                advantages.var().unsqueeze(0),
                eps=self.eps)
            advantages = advantages.squeeze()

        if self.positive_adv:
            advantages -= advantages.min()

        log_likelihood = self.policy.get_log_likelihood(
            torch.Tensor(path['observations']), torch.Tensor(path['actions']))

        grad = (log_likelihood * advantages).sum()

        if self.entropy_regularzied:
            grad += self.policy_ent_coeff * policy_entropy

        return -grad.unsqueeze(0)  # Maximize

    def _get_policy_entropy(self, obs):
        policy_entropy = self.policy.get_entropy(obs).sum()

        if self.stop_entropy_gradient:
            policy_entropy.requires_grad = False

        # This prevents entropy from becoming negative for small policy std
        if self.use_softplus_entropy:
            policy_entropy = F.softplus(policy_entropy)

        return policy_entropy

    def _compute_advantages(self, baselines, rewards):
        path_baselines = F.pad(baselines, (0, 1), value=0)
        deltas = (
            rewards + self.discount * path_baselines[1:] - path_baselines[:-1])

        advantage = special.discount_cumsum(deltas.detach().numpy(),
                                            self.discount * self.gae_lambda)

        return torch.Tensor(advantage.copy())

    def _get_baselines(self, path):
        if hasattr(self.baseline, 'predict_n'):
            return torch.Tensor(self.baseline.predict_n(path))
        else:
            return torch.Tensor(self.baseline.predict(path))

    def _optimize_baseline(self, itr, paths):
        max_path_length = self.max_path_length

        for idx, path in enumerate(paths):
            path['returns'] = special.discount_cumsum(path['rewards'],
                                                      self.discount)

        valids = [np.ones_like(path['returns']) for path in paths]
        valids = tensor_utils.pad_tensor_n(valids, max_path_length)

        rewards = self._collect_element_with_padding(paths, 'rewards',
                                                     max_path_length)
        rewards = [
            rew[val.astype(np.bool)] for rew, val in zip(rewards, valids)
        ]

        returns = self._collect_element_with_padding(paths, 'returns',
                                                     max_path_length)
        returns = [
            ret[val.astype(np.bool)] for ret, val in zip(returns, valids)
        ]

        agent_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(path['agent_infos'], max_path_length)
            for path in paths
        ])

        env_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(path['env_infos'], max_path_length)
            for path in paths
        ])

        samples_data = dict(
            observations=self._collect_element_with_padding(
                paths, 'observations', max_path_length),
            actions=self._collect_element_with_padding(paths, 'actions',
                                                       max_path_length),
            baselines=self._collect_element_with_padding(
                paths, 'baselines', max_path_length),
            rewards=rewards,
            returns=returns,
            valids=valids,
            agent_infos=agent_infos,
            env_infos=env_infos,
            average_return=np.mean([sum(path['rewards']) for path in paths]),
            paths=paths,
        )

        if hasattr(self.baseline, 'fit_with_samples'):
            self.baseline.fit_with_samples(paths, samples_data)
        else:
            self.baseline.fit(paths)

        self._log(itr, paths, valids)

    def _log(self, itr, paths, valids):
        undiscounted_returns = [sum(path['rewards']) for path in paths]
        baselines = [sum(path['baselines']) for path in paths]

        self.reward_mean.extend(undiscounted_returns)

        tabular.record('Iteration', itr)
        tabular.record('AverageDiscountedReturn',
                       (np.mean([path['returns'][0] for path in paths])))
        tabular.record('AverageReturn', np.mean(undiscounted_returns))
        tabular.record('Extras/EpisodeRewardMean', np.mean(self.reward_mean))
        tabular.record('NumTrajs', len(paths))
        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))

        tabular.record('AverageBaselines', np.mean(baselines))
        tabular.record('MaxBaselines', np.max(baselines))
        tabular.record('MinBaselines', np.min(baselines))

    def _collect_element_with_padding(self, paths, key, max_length):
        return tensor_utils.pad_tensor_n([path[key] for path in paths],
                                         max_length)

    def train(self, runner, batch_size):
        """Obtain samplers and start actual training for each epoch."""
        last_return = None

        for epoch in runner.step_epochs():
            runner.step_path = runner.obtain_samples(runner.step_itr,
                                                     batch_size)
            last_return = self.train_once(runner.step_itr, runner.step_path)
            runner.step_itr += 1

        return last_return
