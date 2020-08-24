# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import numpy as np
import os

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import get_activation_fn
from ray.rllib.utils import try_import_torch
from ray.rllib.models import ModelCatalog

torch, nn = try_import_torch()

import torch.nn.functional as F

logger = logging.getLogger(__name__)

class AppendLogStd(nn.Module):
    '''
    An appending layer for free_log_std.
    '''
    def __init__(self, type, init_val, dim):
        super().__init__()
        self.type = type
        self.init_val = init_val

        if self.type=="constant":
            self.log_std = torch.as_tensor([init_val] * dim)
        elif self.type=="state_independent":
            self.log_std = torch.nn.Parameter(
                torch.as_tensor([init_val] * dim))
            self.register_parameter("log_std", self.log_std)
        else:
            raise NotImplementedError
    def forward(self, x):
        assert x.shape[-1] == self.log_std.shape[-1]
        
        shape = list(x.shape)
        for i in range(0, len(shape)-1):
            shape[i] = 1
        log_std = torch.reshape(self.log_std, shape)
        shape = list(x.shape)
        shape[-1] = 1
        log_std = log_std.repeat(shape)

        out = torch.cat([x, log_std], axis=-1)
        return out

class FC(nn.Module):
    ''' 
    A network with fully connected layers.
    '''
    def __init__(self, size_in, size_out, hiddens, activations, 
                 init_weights, append_log_std=False,
                 log_std_type='constant', sample_std=1.0):
        super().__init__()
        layers = []
        prev_layer_size = size_in
        for i, size_hidden in enumerate(hiddens+[size_out]):
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size_hidden,
                    initializer=normc_initializer(
                        init_weights[i]),
                    activation_fn=get_activation_fn(
                        activations[i], framework="torch")))
            prev_layer_size = size_hidden

        if append_log_std:
            layers.append(AppendLogStd(
                type=log_std_type, 
                init_val=np.log(sample_std), 
                dim=size_out))

        self._model = nn.Sequential(*layers)
    def forward(self, x):
        return self._model(x)

class FullyConnectedPolicy(TorchModelV2, nn.Module):
    ''' 
    A policy that generates action and value with FCNN
    '''
    DEFAULT_CONFIG = {
        "log_std_type": "constant",
        "sample_std": 1.0,
        "policy_fn_hiddens": [128, 128],
        "policy_fn_activations": ["relu", "relu", None],
        "policy_fn_init_weights": [1.0, 1.0, 0.01],
        "value_fn_hiddens": [128, 128],
        "value_fn_activations": ["relu", "relu", None],
        "value_fn_init_weights": [1.0, 1.0, 0.01],
    }
    """Generic fully connected network."""
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **model_kwargs):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        ''' Load and check configuarations '''

        assert num_outputs % 2 == 0, (
            "num_outputs must be divisible by two", num_outputs)
        num_outputs = num_outputs // 2

        custom_model_config = FullyConnectedPolicy.DEFAULT_CONFIG.copy()
        custom_model_config_by_user = model_config.get("custom_model_config")
        if custom_model_config_by_user:
            custom_model_config.update(custom_model_config_by_user)

        log_std_type = custom_model_config.get("log_std_type")
        assert log_std_type in ["constant", "state_independent"]

        sample_std = custom_model_config.get("sample_std")
        assert sample_std > 0.0, "The value shoulde be positive"

        policy_fn_hiddens = custom_model_config.get("policy_fn_hiddens")
        policy_fn_activations = custom_model_config.get("policy_fn_activations")
        policy_fn_init_weights = custom_model_config.get("policy_fn_init_weights")

        assert len(policy_fn_hiddens) > 0
        assert len(policy_fn_hiddens)+1 == len(policy_fn_activations)
        assert len(policy_fn_hiddens)+1 == len(policy_fn_init_weights)

        value_fn_hiddens = custom_model_config.get("value_fn_hiddens")
        value_fn_activations = custom_model_config.get("value_fn_activations")
        value_fn_init_weights = custom_model_config.get("value_fn_init_weights")

        assert len(value_fn_hiddens) > 0
        assert len(value_fn_hiddens)+1 == len(value_fn_activations)
        assert len(value_fn_hiddens)+1 == len(value_fn_init_weights)

        dim_state = int(np.product(obs_space.shape))

        ''' Construct the policy function '''

        self._policy_fn = FC(
            size_in=dim_state, 
            size_out=num_outputs, 
            hiddens=policy_fn_hiddens, 
            activations=policy_fn_activations, 
            init_weights=policy_fn_init_weights, 
            append_log_std=True,
            log_std_type=log_std_type, 
            sample_std=sample_std)

        ''' Construct the value function '''

        self._value_fn = FC(
            size_in=dim_state, 
            size_out=1, 
            hiddens=value_fn_hiddens, 
            activations=value_fn_activations, 
            init_weights=value_fn_init_weights, 
            append_log_std=False)

        self._cur_value = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        obs = obs.reshape(obs.shape[0], -1)
        
        logits = self._policy_fn(obs)
        self._cur_value = self._value_fn(obs).squeeze(1)
        
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def save_policy_weights(self, file):
        torch.save(self._policy_fn.state_dict(), file)

class MOEPolicyBase(TorchModelV2, nn.Module):
    ''' 
    A base policy with Mixture-of-Experts structure
    '''
    DEFAULT_CONFIG = {
        "log_std_type": "constant",
        "sample_std": 1.0,
        "expert_hiddens": [
            [128, 128],
            [128, 128],
            [128, 128],
        ],
        "expert_activations": [
            ["relu", "relu", None],
            ["relu", "relu", None],
            ["relu", "relu", None],
        ],
        "expert_init_weights": [
            [1.0, 1.0, 0.01],
            [1.0, 1.0, 0.01],
            [1.0, 1.0, 0.01],
        ],
        "expert_log_std_types": [
            'constant',
            'constant',
            'constant',
        ],
        "expert_sample_stds": [
            0.1,
            0.1,
            0.1,
        ],
        "expert_checkpoints": [
            None,
            None,
            None,
        ],
        "expert_learnable": [
            True,
            True,
            True,
        ],

        "gate_fn_hiddens": [128, 128],
        "gate_fn_activations": ["relu", "relu", None],
        "gate_fn_init_weights": [1.0, 1.0, 0.01],
        "gate_fn_learnable": True,
        
        "value_fn_hiddens": [128, 128],
        "value_fn_activations": ["relu", "relu", None],
        "value_fn_init_weights": [1.0, 1.0, 0.01],
    }
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **model_kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        ''' Load and check configuarations '''

        assert num_outputs % 2 == 0, (
            "num_outputs must be divisible by two", num_outputs)
        num_outputs = num_outputs // 2

        custom_model_config = MOEPolicyBase.DEFAULT_CONFIG.copy()
        custom_model_config_by_user = model_config.get("custom_model_config")
        if custom_model_config_by_user:
            custom_model_config.update(custom_model_config_by_user)

        log_std_type = custom_model_config.get("log_std_type")
        assert log_std_type in ["constant", "state_independent"]

        sample_std = custom_model_config.get("sample_std")
        assert sample_std > 0.0, "The value shoulde be positive"

        expert_hiddens = custom_model_config.get("expert_hiddens")
        expert_activations = custom_model_config.get("expert_activations")
        expert_init_weights = custom_model_config.get("expert_init_weights")
        expert_log_std_types = custom_model_config.get("expert_log_std_types")
        expert_sample_stds = custom_model_config.get("expert_sample_stds")
        expert_checkpoints = custom_model_config.get("expert_checkpoints")
        expert_learnable = custom_model_config.get("expert_learnable")

        gate_fn_hiddens = custom_model_config.get("gate_fn_hiddens")
        gate_fn_activations = custom_model_config.get("gate_fn_activations")
        gate_fn_init_weights = custom_model_config.get("gate_fn_init_weights")

        value_fn_hiddens = custom_model_config.get("value_fn_hiddens")
        value_fn_activations = custom_model_config.get("value_fn_activations")
        value_fn_init_weights = custom_model_config.get("value_fn_init_weights")

        dim_state = int(np.product(obs_space.shape))
        num_experts = len(expert_hiddens)

        ''' Construct the gate function '''
        
        self._gate_fn = FC(
            size_in=dim_state, 
            size_out=num_experts, 
            hiddens=gate_fn_hiddens, 
            activations=gate_fn_activations, 
            init_weights=gate_fn_init_weights, 
            append_log_std=False)

        ''' Construct experts '''

        self._experts = []
        for i in range(num_experts):
            expert = FC(
                size_in=dim_state, 
                size_out=num_outputs, 
                hiddens=expert_hiddens[i], 
                activations=expert_activations[i], 
                init_weights=expert_init_weights[i], 
                append_log_std=True,
                log_std_type=expert_log_std_types[i], 
                sample_std=expert_sample_stds[i])
            if expert_checkpoints[i]:
                expert.load_state_dict(torch.load(expert_checkpoints[i]))
                expert.eval()
            for name, param in expert.named_parameters():
                param.requires_grad = expert_learnable[i]
            self._experts.append(expert)

        ''' Construct the value function '''
        
        self._value_fn = FC(
            size_in=dim_state, 
            size_out=1, 
            hiddens=value_fn_hiddens, 
            activations=value_fn_activations, 
            init_weights=value_fn_init_weights, 
            append_log_std=False)

        self._num_experts = num_experts
        self._cur_value = None
        self._cur_gate_weight = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        raise NotImplementedError

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def gate_function(self):
        return self._cur_gate_weight

    def num_experts(self):
        return self._num_experts

class MOEPolicyAdditive(MOEPolicyBase):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **model_kwargs):
        super().__init__(
            obs_space, action_space, num_outputs, 
            model_config, name, **model_kwargs)
    
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        obs = obs.reshape(obs.shape[0], -1)
        
        w = F.softmax(self._gate_fn(obs), dim=1)
        x = 0.0
        for i, expert in enumerate(self._experts):
            x += w[...,i]*expert(obs)
        logits = x

        self._cur_gate_weight = w
        self._cur_value = self._value_fn(obs).squeeze(1)
        
        return logits, state

class MOEPolicyMultiplicative(MOEPolicyBase):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **model_kwargs):
        super().__init__(
            obs_space, action_space, num_outputs, 
            model_config, name, **model_kwargs)
    
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        obs = obs.reshape(obs.shape[0], -1)
        
        w = F.softmax(self._gate_fn(obs), dim=1).unsqueeze(-1)
        x = torch.stack([expert(obs) for expert in self._experts], dim=1)

        expert_mean = x[...,:self.num_outputs]
        expert_std = torch.exp(x[...,self.num_outputs:])

        z = w / expert_std
        std = 1.0 / torch.sum(z, dim=1)
        logstd = torch.log(std)
        mean = std * torch.sum(z * expert_mean, dim=1)
        
        logits = torch.concat([], )

        self._cur_weight = w
        self._cur_value = self._value_fn(obs).squeeze(1)
        
        return logits, state

ModelCatalog.register_custom_model("fcnn", FullyConnectedPolicy)
ModelCatalog.register_custom_model("moe_additive", MOEPolicyAdditive)
ModelCatalog.register_custom_model("moe_multiplicative", MOEPolicyMultiplicative)