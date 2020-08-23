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
        "log_std_type": "state_independent",
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
        "log_std_type": "state_independent",
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
            'state_independent',
            'state_independent',
            'state_independent',
        ],
        "expert_sample_stds": [
            0.2,
            0.2,
            0.2,
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

        # # Batch sytle inference
        # w = F.softmax(self._gate_fn(obs), dim=1).unsqueeze(-1)
        # x = torch.stack([expert(obs) for expert in self._experts], dim=1)
        # logits = torch.sum(w*x, dim=1)
        
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

class TaskAgnosticPolicyType1(TorchModelV2, nn.Module):
# class TaskAgnosticPolicyType1(RecurrentTorchModel, nn.Module):
    DEFAULT_CONFIG = {
        "project_dir": None,
        
        "log_std_type": "constant",
        "sample_std": 1.0,
        
        "lstm_enable": False,
        "lstm_hidden_size": 32,
        "lstm_num_layers": 2,
        
        "motor_decoder_hiddens": [128, 128],
        "motor_decoder_activation": "relu",
        "motor_decoder_weights": None,
        "motor_decoder_learnable": True,
        
        "task_encoder_enable": True,
        "task_encoder_hiddens": [128, 128],
        "task_encoder_activation": "relu",
        "task_encoder_output_dim": 32,
        
        "body_encoder_enable": False,
        "body_encoder_hiddens": [128, 128],
        "body_encoder_activation": "relu",
        "body_encoder_output_dim": 32,
        "body_encoder_weights": None,
        "body_encoder_learnable": True,
        
        "value_fn_hiddens": [128, 128],
        "value_fn_activation": "relu",
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

        custom_model_config = TaskAgnosticPolicyType1.DEFAULT_CONFIG.copy()
        custom_model_config_by_user = model_config.get("custom_model_config")
        if custom_model_config_by_user:
            custom_model_config.update(custom_model_config_by_user)

        log_std_type = custom_model_config.get("log_std_type")
        if log_std_type is None:
            sample_std = TaskAgnosticPolicyType1.DEFAULT_CONFIG["log_std_type"]
        assert log_std_type in ["constant", "state_independent"]

        sample_std = custom_model_config.get("sample_std")
        if sample_std is None:
            sample_std = TaskAgnosticPolicyType1.DEFAULT_CONFIG["sample_std"]

        project_dir = custom_model_config.get("project_dir")

        task_encoder_enable = custom_model_config.get("task_encoder_enable")
        task_encoder_hiddens = custom_model_config.get("task_encoder_hiddens")
        task_encoder_activation = custom_model_config.get("task_encoder_activation")
        task_encoder_output_dim = custom_model_config.get("task_encoder_output_dim")

        body_encoder_enable = custom_model_config.get("body_encoder_enable")
        body_encoder_hiddens = custom_model_config.get("body_encoder_hiddens")
        body_encoder_activation = custom_model_config.get("body_encoder_activation")
        body_encoder_output_dim = custom_model_config.get("body_encoder_output_dim")
        body_encoder_weights = custom_model_config.get("body_encoder_weights")
        body_encoder_learnable = custom_model_config.get("body_encoder_learnable")

        motor_decoder_hiddens = custom_model_config.get("motor_decoder_hiddens")
        motor_decoder_activation = custom_model_config.get("motor_decoder_activation")
        motor_decoder_weights = custom_model_config.get("motor_decoder_weights")
        motor_decoder_learnable = custom_model_config.get("motor_decoder_learnable")

        value_fn_hiddens = custom_model_config.get("value_fn_hiddens")
        value_fn_activation = custom_model_config.get("value_fn_activation")

        lstm_enable = custom_model_config.get("lstm_enable")
        lstm_hidden_size = custom_model_config.get("lstm_hidden_size")
        lstm_num_layers = custom_model_config.get("lstm_num_layers")

        if project_dir:
            if body_encoder_weights:
                body_encoder_weights = \
                    os.path.join(project_dir, body_encoder_weights)
                assert body_encoder_weights
            if motor_decoder_weights:
                motor_decoder_weights = \
                    os.path.join(project_dir, motor_decoder_weights)
                assert motor_decoder_weights

        self.dim_state_body = int(np.product(custom_model_config.get("observation_space_body").shape))
        self.dim_state_task = int(np.product(custom_model_config.get("observation_space_task").shape))
        self.dim_state = int(np.product(obs_space.shape))

        assert self.dim_state == self.dim_state_body + self.dim_state_task

        ''' Prepare task encoder that outputs task embedding z given s_task '''

        if task_encoder_enable:
            activation = get_activation_fn(task_encoder_activation, framework="torch")
            layers = []
            prev_layer_size = self.dim_state_task

            for size in task_encoder_hiddens:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=size,
                        initializer=normc_initializer(1.0),
                        activation_fn=activation))
                prev_layer_size = size
            # Output of the task encoder will be confined by tanh
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=task_encoder_output_dim,
                    initializer=normc_initializer(0.01),
                    activation_fn=nn.Tanh))
            self._task_encoder = nn.Sequential(*layers)
        else:
            self._task_encoder = None

        ''' Prepare body encoder that outputs body embedding z given s_task '''

        if body_encoder_enable:
            activation = get_activation_fn(body_encoder_activation, framework="torch")
            layers = []
            prev_layer_size = self.dim_state_body

            for size in body_encoder_hiddens:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=size,
                        initializer=normc_initializer(1.0),
                        activation_fn=activation))
                prev_layer_size = size
            # Output of the task encoder will be confined by tanh
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=body_encoder_output_dim,
                    initializer=normc_initializer(1.0),
                    activation_fn=nn.Tanh))
            self._body_encoder = nn.Sequential(*layers)
        else:
            self._body_encoder = None

        dim_state_body = body_encoder_output_dim if self._body_encoder else self.dim_state_body
        dim_state_task = task_encoder_output_dim if self._task_encoder else self.dim_state_task

        if lstm_enable:
            input_size = dim_state_body + dim_state_task
            self._lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                batch_first=True)
            self._lstm_project_output = nn.Linear(lstm_hidden_size, input_size)
            self.lstm_hidden_size = lstm_hidden_size
            self.lstm_num_layers = lstm_num_layers
            self.lstm_input_size = input_size
        else:
            self._lstm = None

        ''' Prepare motor control decoder that outputs a given (z, s_proprioception) '''

        activation = get_activation_fn(motor_decoder_activation, framework="torch")
        layers = []
        prev_layer_size = dim_state_body + dim_state_task

        for size in motor_decoder_hiddens:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation))
            prev_layer_size = size
        layers.append(
            SlimFC(
                in_size=prev_layer_size,
                out_size=num_outputs,
                initializer=normc_initializer(0.01),
                activation_fn=None))

        layers.append(AppendLogStd(
                type=log_std_type, 
                init_val=np.log(sample_std), 
                dim=num_outputs))

        self._motor_decoder = nn.Sequential(*layers)

        if motor_decoder_weights:
            self.load_weights_motor_decoder(motor_decoder_weights)
            for name, param in self._motor_decoder.named_parameters():
                param.requires_grad = True if motor_decoder_learnable else False
                # Let log_std always be learnable
                if 'log_std' in name:
                    param.requires_grad = True 

        if body_encoder_weights:
            self.load_weights_body_encoder(body_encoder_weights)
            for name, param in self._body_encoder.named_parameters():
                param.requires_grad = True if body_encoder_learnable else False

        ''' Prepare a value function '''

        activation = get_activation_fn(value_fn_activation, framework="torch")
        layers = []
        prev_layer_size = self.dim_state
        for size in value_fn_hiddens:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation))
            prev_layer_size = size
        layers.append(
            SlimFC(
                in_size=prev_layer_size,
                out_size=1,
                initializer=normc_initializer(0.01),
                activation_fn=None))

        self._value_branch = nn.Sequential(*layers)

        self._cur_value = None

    @override(TorchModelV2)
    def get_initial_state(self):
        if self._lstm:
            # The shape should be (num_hidden_layers, hidden_size)
            h0 = self._motor_decoder[0]._model[0].weight.new(
                self.lstm_num_layers, self.lstm_hidden_size).zero_()
            c0 = self._motor_decoder[0]._model[0].weight.new(
                self.lstm_num_layers, self.lstm_hidden_size).zero_()
            return [h0, c0]
        else:
            return []

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        ''' Assume state==(state_body, state_task) '''
        
        obs = input_dict["obs_flat"].float()
        
        obs_body = obs[...,:self.dim_state_body]
        obs_task = obs[...,self.dim_state_body:]

        z_body = self._body_encoder(obs_body) if self._body_encoder else obs_body
        z_task = self._task_encoder(obs_task) if self._task_encoder else obs_task
        z = torch.cat([z_body, z_task], axis=-1)

        if self._lstm:
            if isinstance(seq_lens, np.ndarray):
                seq_lens = torch.Tensor(seq_lens).int()
            z = add_time_dimension(z, seq_lens, framework="torch")

            ''' 
            The shape of the hidden states should be 
            (num_hidden_layers, batch_size, hidden_size) in PyTorch
            '''

            h_lstm, c_lstm = state[0], state[1]
            h_lstm = h_lstm.reshape(h_lstm.shape[1], h_lstm.shape[0], h_lstm.shape[2])
            c_lstm = c_lstm.reshape(c_lstm.shape[1], c_lstm.shape[0], c_lstm.shape[2])
            
            output_lstm, (h_lstm, c_lstm) = self._lstm(z, (h_lstm, c_lstm))
            z = self._lstm_project_output(output_lstm)
            z = z.reshape(-1, z.shape[-1])
            
            h_lstm = h_lstm.reshape(h_lstm.shape[1], h_lstm.shape[0], h_lstm.shape[2])
            c_lstm = c_lstm.reshape(c_lstm.shape[1], c_lstm.shape[0], c_lstm.shape[2])
            
            state = [h_lstm, c_lstm]
            
        logits = self._motor_decoder(z)
        self._cur_value = self._value_branch(obs).squeeze(1)
        
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def save_weights_body_encoder(self, file):
        assert self._body_encoder
        torch.save(self._body_encoder.state_dict(), file)

    def load_weights_body_encoder(self, file):
        assert self._body_encoder
        self._body_encoder.load_state_dict(torch.load(file))
        self._body_encoder.eval()

    def save_weights_motor_decoder(self, file):
        torch.save(self._motor_decoder.state_dict(), file)

    def load_weights_motor_decoder(self, file):
        ''' Ignore weights of log_std for valid exploration '''
        dict_weights_orig = self._motor_decoder.state_dict()
        dict_weights_loaded = torch.load(file)
        for key in dict_weights_loaded.keys():
            if 'log_std' in key:
                dict_weights_loaded[key] = dict_weights_orig[key]
                # print(dict_weights_orig[key])
        self._motor_decoder.load_state_dict(dict_weights_loaded)
        self._motor_decoder.eval()

ModelCatalog.register_custom_model("fcnn", FullyConnectedPolicy)
ModelCatalog.register_custom_model("moe_additive", MOEPolicyAdditive)
ModelCatalog.register_custom_model("task_agnostic_policy_type1", TaskAgnosticPolicyType1)