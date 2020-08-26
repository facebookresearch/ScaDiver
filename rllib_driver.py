# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import os
import yaml

import ray

from ray import tune
from ray.tune.registry import register_env

def arg_parser():
    parser = argparse.ArgumentParser()
    ''' Specification file of the expriment '''
    parser.add_argument("--spec", required=True, type=str)
    ''' Mode for running an experiment '''
    parser.add_argument("--mode", required=True, choices=['train', 'load'])
    '''  '''
    parser.add_argument("--checkpoint", type=str, default=None)
    '''  '''
    parser.add_argument("--num_workers", type=int, default=None)
    '''  '''
    parser.add_argument("--num_cpus", type=int, default=1)
    '''  '''
    parser.add_argument("--num_gpus", type=int, default=0)
    '''  '''
    parser.add_argument("--num_envs_per_worker", type=int, default=None)
    '''  '''
    parser.add_argument("--num_cpus_per_worker", type=int, default=None)
    '''  '''
    parser.add_argument("--num_gpus_per_worker", type=int, default=None)
    ''' Directory where the environment and related files are stored '''
    parser.add_argument("--project_dir", type=str, default=None)
    ''' Directory where intermediate results are saved '''
    parser.add_argument("--local_dir", type=str, default=None)
    ''' Verbose '''
    parser.add_argument("--verbose", action='store_true')
    '''  '''
    parser.add_argument("--ip_head", type=str, default=None)
    '''  '''
    parser.add_argument("--password", type=str, default=None)

    return parser

if __name__ == "__main__":

    args = arg_parser().parse_args()
    
    with open(args.spec) as f:
        spec = yaml.load(f, Loader=yaml.FullLoader)

    config = spec['config']

    '''
    Register environment to learn according to the input specification file
    '''

    if config['env'] == "HumanoidImitation":
        import rllib_env_imitation as env_module
    else:
        raise NotImplementedError("Unknown Environment")

    register_env(config['env'], lambda config: env_module.env_cls(config))

    '''
    Register custom model to use if it exists
    '''

    framework = config.get('framework')

    if config.get('model'):
        custom_model = config.get('model').get('custom_model')
        if custom_model:
            if framework=='torch':
                import rllib_model_custom_torch
            else:
                raise NotImplementedError("Tensorflow is not supported!")

    '''
    Validate configurations and overide values by arguments
    '''

    if args.local_dir is not None:
        spec.update({'local_dir': args.local_dir})
    
    if args.project_dir is not None:
        assert os.path.exists(args.project_dir)
        config['env_config']['project_dir'] = args.project_dir
    
    if config['model'].get('custom_model_config'):
        config['model']['custom_model_config'].update(
            {'project_dir': config['env_config']['project_dir']})

    if args.verbose:
        config['env_config'].update({'verbose': args.verbose})

    if args.checkpoint is not None:
        assert os.path.exists(args.checkpoint)
    
    if args.num_workers is not None:
        config.update({'num_workers': args.num_workers})
    
    if args.num_gpus is not None:
        config.update({'num_gpus': args.num_gpus})

    if args.num_envs_per_worker:
        config.update({'num_envs_per_worker': args.num_envs_per_worker})

    if args.num_cpus_per_worker:
        config.update({'num_cpus_per_worker': args.num_cpus_per_worker})

    if args.num_gpus_per_worker:
        config.update({'num_gpus_per_worker': args.num_gpus_per_worker})

    if args.mode == "train":
        if not os.path.exists(spec['local_dir']):
            raise Exception(
                "The directory does not exist: %s"%spec['local_dir'])

    config_override = env_module.config_override(spec)
    config.update(config_override)

    if args.ip_head:
        # tmp_dir = os.path.join(spec['local_dir'], os.path.join('tmp/', spec['name']))
        if args.password:
            ray.init(address=args.ip_head, redis_password=args.password)
        else:
            ray.init(address=args.ip_head)
    else:
        assert args.num_cpus is not None
        assert args.num_gpus is not None
        ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus)

    def adjust_config_for_loading(config, alg):
        config["num_workers"] = 1

        config['num_envs_per_worker'] = 1
        config['num_cpus_per_worker'] = 1
        config['num_gpus_per_worker'] = 0
        config['remote_worker_envs'] = False

    def adjust_config(config, alg):
        rollout_fragment_length = config.get('rollout_fragment_length')
        num_workers = config.get('num_workers')
        num_envs_per_worker = config.get('num_envs_per_worker')
        train_batch_size = config.get('train_batch_size')

        ''' 
        Set rollout_fragment_length value so that
        workers can genertate train_batch_size tuples correctly
        '''
        rollout_fragment_length = \
            max(train_batch_size // (num_workers * num_envs_per_worker), 100)
    
        while rollout_fragment_length * num_workers * num_envs_per_worker \
            < train_batch_size:
            rollout_fragment_length += 1

        config['rollout_fragment_length'] = rollout_fragment_length

    adjust_config(config, spec['run'])
    
    if args.mode == "load":
        adjust_config_for_loading(config, spec['run'])

        if spec["run"] == "PPO":
            from ray.rllib.agents.ppo import PPOTrainer as Trainer
        else:
            raise NotImplementedError("Not a supported algorithm")

        trainer = Trainer(env=env_module.env_cls, config=config)

        if args.checkpoint is not None:
            trainer.restore(args.checkpoint)

        env_module.rm.initialize()

        env = env_module.env_cls(config['env_config'])
        cam = env_module.default_cam()

        renderer = env_module.EnvRenderer(trainer=trainer, env=env, cam=cam)
        renderer.run()
    else:
        tune.run(
            spec['run'],
            name=spec['name'],
            stop=spec['stop'],
            local_dir=spec['local_dir'],
            checkpoint_freq=spec['checkpoint_freq'],
            checkpoint_at_end=spec['checkpoint_at_end'],
            config=config,
            restore=args.checkpoint,
            sync_to_driver=False,
        )
