# ScaDiver

## Introduction
This repository provides excutable codes for the paper *A Scalable Approach to Control Diverse Behaviors for Physically Simulated Characters*, which was published in SIGGRAPH 2020. Click [here](https://research.fb.com/publications/a-scalable-approach-to-control-diverse-behaviors-for-physically-simulated-characters/) to see the paper. The name of the project originates from the two keywords in the name of the paper, which are "**sca**lable" and "**diver**se".

## Citation
```
@article{
    ScaDiver,
    author = {Won, Jungdam and Gopinath, Deepak and Hodgins, Jessica},
    title = {A Scalable Approach to Control Diverse Behaviors for Physically Simulated Characters},
    year = {2020},
    issue_date = {July 2020},
    volume = {39},
    number = {4},
    url = {https://doi.org/10.1145/3386569.3392381},
    journal = {ACM Trans. Graph.},
    articleno = {33},
}
```

## Getting Started

### Installation

[fairmotion](https://github.com/fairinternal/fairmotion) provides functionalities to process moion capture data and, to compute kinematics, to visualize simulated characters and environments.

#### fairmotion
```
pip install fairmotion
```

We use [PyBullet](https://pybullet.org/wordpress/) for physics simulation and [rllib](https://docs.ray.io/en/latest/rllib.html) for reinforcement learning. 

#### others
```
pip install pybullet==2.7.3 ray[rllib]==0.8.6 pandas requests
```

### Examples

#### Test Tracking Environment

In this example, a humanoid character will be loaded into a empty space where only the ground plane exists, and will be simulated without any control. Please press **a** to simulate and press **r** to reset.

```
python3 env_humanoid_tracking.py
```

#### Test Imitation Environment

In this example, a humanoid character will be loaded into a empty space where only the ground plane exists, and its initial states will be set by some motion capture data specified in the specification file (test_env_humanoid_imiation.yaml). Please note that the character will fall immediately because the current controller is the one not trained at all.

```
python3 rllib_driver.py --mode load --spec data/spec/test_env_humanoid_imitation.yaml
```

#### Learning Experts
```
...
```

#### Learning Mixture of Experts
```
...
```

### Specification File

Every experiment requries an unique specification file (yaml) that includes the all the setting about individual expriements. Most of elements are easy to understand by its name and it is based on **rllib**'s configuration file. We will explain only our environment-specific settings below, please refer to the documentation of **rllib** for other settings.

#### Reward

Because the optimality of learned policy is defined by the definition of reward function in reinforcement learning, changing the function is the only way for users to design the policy so that it has desirable properties/behaviors. As a result, testing many reward functions by combining various terms with different combinations is a critical process in many researches including ours. We implemented a flexible way to test various functions through simply modifying a specification file.

Our reward function is defined in a tree-like manner, where each node could be a *operation* among child nodes or a *leaf* that defines a term. For example, the funcion below is a multiplicative reward function composed of the five terms (pose_pos, pose_vel, ee, root, and com), where each term has its own kernel function. Currently, we only support *gaussian* and *none* kernel functions. By simply changing the operation of the root node from *mul* into *add*, we can test an additive reward function. If we want to change weight values for the terms, we can simply change *weight* in the leaf nodes.

We can also define multiple reward functions in the same specification files (please note that their names should be unique), then choose one of them according to experiments by setting the name which we want to test in *fn_map*.

```
reward: 
    fn_def:
        default:
            name: total
            op: mul
            child_nodes:
              - name: pose_pos
                op: leaf
                weight: 1.0
                kernel: 
                    type: gaussian
                    scale: 40.0
              - name: pose_vel
                op: leaf
                weight: 1.0
                kernel: 
                    type: gaussian
                    scale: 1.0
              - name: ee
                op: leaf
                weight: 1.0
                kernel: 
                    type: gaussian
                    scale: 5.0
              - name: root
                op: leaf
                weight: 1.0
                kernel: 
                    type: gaussian
                    scale: 2.5
              - name: com
                op: leaf
                weight: 1.0
                kernel: 
                    type: gaussian
                    scale: 2.5
    fn_map:
        - default
```

#### Early Termination

We can use vairous early termination stratigies. For example, we can terminate the current episode when the character falls down, or the average reward for 1s is below a specific threshold as used in our paper. 

```
early_term:
    choices: # 'sim_div', 'sim_window', task_end', 'falldown', 'low_reward'
        - task_end
        - low_reward
    low_reward_thres: 0.1
    eoe_margin: 0.5
```

#### Characters

The character fields include character-specific information. 

```
character:
    name:
        - humanoid
    char_info_module:
        - amass_char_info.py
    sim_char_file:
        - data/character/amass.urdf
    ref_motion_scale:
        - 1.0
    base_motion_file:
        - data/motion/amass_hierarchy.bvh
    ref_motion_db:
        -
            data:
                file:
                    - data/motion/test.bvh
    actuation: 
        - spd
    self_collision: 
        - true
    environment_file: []
```

#### Others

If true, the environment will be fully created when *reset* is called. This is useful when the creation of environment is expensive. Please search with the keyword *Expensive Environments* [here](https://docs.ray.io/en/latest/rllib-env.html#:~:text=Expensive%20Environments,-Some%20environments%20may&text=RLlib%20will%20create%20num_workers%20%2B%201,until%20reset()%20is%20called.) for more details.

```
lazy_creation: false
```

```
project_dir: /home/jungdam/Research/opensource/ScaDiver/
```

Time steps for physics simulation

```
fps_sim: 480
```

Time steps for control policies

```
fps_con: 30
```

If true, small noise will be added to th the initial state of the character
```
add_noise: false
```

If true, some additional informations will be printed. Please use this only for testing environments.

```
verbose: false
```

This defines which information will be added to the state of the character.

```
state:
    choices: ['body', 'task'] # 'body', 'imitation', 'interaction', 'task'
```

This defines the action space for our environment.

```
action:
    type: "absolute" # 'absolute', 'relative'
    range_min: -3.0
    range_max: 3.0
    range_min_pol: -3.0
    range_max_pol: 3.0
```