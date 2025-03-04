# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the Lagrangian version of Soft Actor-Critic algorithm."""


import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.models.actor_critic.constraint_actor_q_critic import DoubleConstraintActorQCritic
from omnisafe.common.smooth_log_barrier import LinearSmoothedLogBarrier

@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class SACLB(SAC):
    """The Lagrangian version of Soft Actor-Critic (SAC) algorithm.

    References:
        - Title: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
        - Authors: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine.
        - URL: `SAC <https://arxiv.org/abs/1801.01290>`_
    """

    def _init_model(self) -> None:
        """Initialize the model.

        The ``num_critics`` in ``critic`` configuration must be 2.
        """
        self._cfgs.model_cfgs.critic['num_critics'] = 2
        self._actor_critic = DoubleConstraintActorQCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)
        self._log_barrier = LinearSmoothedLogBarrier(**self._cfgs.log_barrier_cfgs)


    def _init(self) -> None:
        """The initialization of the algorithm.

        Here we additionally initialize the Lagrange multiplier.
        """
        super()._init()

    def _init_log(self) -> None:
        """Log the SACLag specific information.

        +-------------------------+----------------------------------------+
        | Things to log           | Description                            |
        +=========================+========================================+
        | Metrics/LB_Offset       | The offset of the log barrier function.|
        +-------------------------+----------------------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LB_Offset')

    def _update(self) -> None:
        """Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the Lagrange multiplier parameter by calling the
        :meth:`update_lagrange_multiplier` method.
        """
        super()._update()
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        self._log_barrier.update_lb_offset(self._epoch, self._epochs)
        self._logger.store(
            {
                'Metrics/LB_Offset': self._log_barrier.offset,
            },
        )

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing ``pi/actor`` loss.

        The loss function in SACLag is defined as:

        .. math::

            L = -Q^V (s, \pi (s)) + \lambda Q^C (s, \pi (s))

        where :math:`Q^V` is the min value of two reward critic networks outputs, :math:`Q^C` is the
        value of cost critic network, and :math:`\pi` is the policy network.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.

        Returns:
            The loss of pi/actor.
        """
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)
        loss_q_r_1, loss_q_r_2 = self._actor_critic.reward_critic(obs, action)
        loss_r = self._alpha * log_prob - torch.min(loss_q_r_1, loss_q_r_2)
        loss_q_c_1, loss_q_c_2 = self._actor_critic.cost_critic(obs, action)
        loss_c = self._log_barrier(torch.max(loss_q_c_1, loss_q_c_2))

        return (loss_r + loss_c).mean()

    def _log_when_not_update(self) -> None:
        super()._log_when_not_update()
        self._logger.store(
            {
                'Metrics/LB_Offset': self._log_barrier.offset,
            },
        )
