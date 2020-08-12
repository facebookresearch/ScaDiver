# ScaDiver

## Introduction
```
This repository provides excutable codes for the paper "A Scalable Approach to Control Diverse Behaviors for Physically Simulated Characters", which was published in SIGGRAPH 2020.
```

## Getting Started

### Installation

#### fairmotion
```
https://github.com/fairinternal/fairmotion
```

#### others
```
pip install pybullet==2.7.3 ray[rllib]==0.8.6 pandas requests
```

### Examples

#### Test Tracking Environment
```
python3 env_humanoid_tracking.py
```

#### Test Imitation Environment
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
