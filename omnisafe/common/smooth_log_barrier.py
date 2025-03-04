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
"""Implementation of Linear Smoothed Log Barrier Function."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import math
class LinearSmoothedLogBarrier:
    """Linear Smoothed Log Barrier Function.

    This class implements the linear smoothed log barrier function

    ..  note::
        The linear smoothed log barrier function provides a smooth approximation to the hard
        constraint by using a logarithmic barrier term that grows as the constraint is approached.

    Examples:
        >>> from omnisafe.common.smooth_log_barrier import LinearSmoothedLogBarrier
        >>> def loss_pi(self, data):
        ...     # implement your own loss function here
        ...     return loss

    You can also inherit this class to implement your own algorithm with smoothed log barrier
    function in OmniSafe.

    Examples:
        >>> from omnisafe.common.smooth_log_barrier import LinearSmoothedLogBarrier
        >>> class CustomAlgo:
        ...     def __init__(self) -> None:
        ...         # initialize your own algorithm here
        ...         super().__init__()
        ...         # initialize the smoothed log barrier
        ...         self.barrier = LinearSmoothedLogBarrier(**self._cfgs.barrier_cfgs)

    Args:
        cost_limit (float): The cost limit.
        log_barrier_factor (float): The factor $t$ of the log barrier function. Defaults to 3.0.
        log_barrier_offset (float): The offset of the log barrier function. Defaults to -1.0.
        log_barrier_eps (float): The epsilon of the log barrier function. Defaults to 1e-7.
        update_mode (str): The update mode of the log barrier function. Must be one of 'const' or 'linear'. Defaults to 'const'.
        update_offset_args (dict): Additional arguments for updating the offset. Defaults to {}.

    Attributes:
        cost_limit (float): The cost limit.
        lb_factor (float): The factor $t$ of the log barrier function.
        offset (float): The offset of the log barrier function.
        eps (float): The epsilon of the log barrier function.
        update_mode (str): The update mode of the log barrier function.
        update_offset_args (dict): Additional arguments for updating the offset.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        cost_limit: float,
        log_barrier_factor: float = 3.0,
        log_barrier_offset: float = -1.0,
        log_barrier_eps: float = 1e-7,
        update_mode: str = 'const',
        update_offset_args: dict = {},
    ) -> None:
        """Initialize an instance of :class:`LinearSmoothedLogBarrier`.
        
        Args:
            cost_limit (float): The cost limit.
            log_barrier_factor (float): The factor $t$ of the log barrier function.
            log_barrier_offset (float): The offset of the log barrier function.
            log_barrier_eps (float): The epsilon of the log barrier function.
            update_mode (str): The update mode of the log barrier function.
            update_offset_args (dict): Additional arguments for updating the offset.
                If update_mode is 'const', update_offset_args is not used.
                If update_mode is 'linear', update_offset_args should contain the following keys:
                    - offset_start (float): The start offset.
                    - offset_end (float): The end offset.
                    - offset_start_ratio (float): The start epoch ratio of the update.
                    - offset_end_ratio (float): The end epoch ratio of the update.
        """
        self.cost_limit: float = cost_limit
        self.update_offset_args: dict = update_offset_args
        self.lb_factor: float = float(log_barrier_factor)
        self.eps: float = log_barrier_eps
        assert update_mode in ['const', 'linear'], 'update_mode must be one of const, linear, exp'
        if self.update_offset_args.get('update_mode', 'const') == 'const':
            self.offset: float = log_barrier_offset
        else:
            self.offset: float = self.update_offset_args.get('offset_start', 0.0)
        self.update_mode: str = update_mode
        if update_mode == 'const':
            self.update_lb_offset = lambda cur_step, max_step: self.update_lb_offset_const(**update_offset_args)
        elif update_mode == 'linear':
            self.update_lb_offset = lambda cur_step, max_step: self.update_lb_offset_linear(cur_step, max_step, **update_offset_args)
        else:
            raise ValueError(f'update_mode must be one of const or linear, but got {update_mode}')
    
    def __call__(self, input: torch.tensor) -> torch.Tensor:
        r"""Compute the linear smoothed log barrier function.

        .. math::
            \phi(x) = \begin{cases}
                \frac{1}{t} \log(1 + t(x - \epsilon)) & \text{if } x \leq \epsilon - \frac{1}{t} \\
                t(x - \frac{1}{t}) - \frac{1}{t} \log(1 + t(\epsilon - x)) & \text{if } x > \epsilon - \frac{1}{t}
            \end{cases}
        """
        effective_cost_limit = self.cost_limit + self.offset
        x = F.relu(input - effective_cost_limit) - 1.0
        return torch.where(x <= -1/self.lb_factor**2,
                           -1/self.lb_factor*torch.log(-x+self.eps),
                           self.lb_factor*x-1/self.lb_factor*math.log(1/self.lb_factor**2)+1/self.lb_factor)

    def update_lb_offset_const(self, **kwargs) -> None:
        r"""Update the log barrier factor.
        """
        pass

    def update_lb_offset_linear(self, cur_epoch: int, max_epoch: int, **kwargs) -> None:
        r"""Update the log barrier factor.
        
        Args:
            cur_step (int): The current update epoch.
            max_step (int): The maximum update epochs.
            offset_start (float): The start offset.
            offset_end (float): The end offset.
            offset_start_ratio (float): The start epoch ratio of the update.
            offset_end_ratio (float): The end epoch ratio of the update.
        """
        offset_start = self.update_offset_args.get('offset_start', 0.0)
        offset_end = self.update_offset_args.get('offset_end', 1.0)
        offset_start_ratio = self.update_offset_args.get('offset_start_ratio', 0.0)
        offset_end_ratio = self.update_offset_args.get('offset_end_ratio', 1.0)

        # Calculate effective step range based on start/end ratios
        start_epoch = int(max_epoch * offset_start_ratio)
        end_epoch = int(max_epoch * offset_end_ratio)
        
        # Only update offset within the specified ratio range
        if cur_epoch < start_epoch:
            self.offset = offset_start
        elif cur_epoch > end_epoch:
            self.offset = offset_end
        else:
            # Linear interpolation within the effective range
            progress = (cur_epoch - start_epoch) / (end_epoch - start_epoch)
            self.offset = offset_start + (offset_end - offset_start) * progress