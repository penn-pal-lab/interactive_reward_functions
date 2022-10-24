# Training Robots to Evaluate Robots: Example-Based Interactive Reward Functions for Policy Learning

This repository contains the source code for our paper:

[Training Robots to Evaluate Robots: Example-Based Interactive Reward Functions for Policy Learning](https://openreview.net/forum?id=sK2aWU7X9b8)\
CoRL 2022\
Kun Huang, Edward S. Hu, Dinesh Jayaraman

## Install Python Environment

### Roboaware

1. Run `conda env create -f environment.yml`, then activate this conda environment.
2. Clone the [`d4rl`](https://github.com/voyager1998/d4rl.git)
3. cd into the d4rl repo, run `pip install -e .`

### ROBEL

WARNING: Do not install robel package inside `Roboaware` conda env, it's incompatible.

1. Clone [ROBEL](https://github.com/voyager1998/robel.git) repo into another folder.
2. Run `pip install -e .` inside robel.
3. For the rest, follow exactly the [instructions](https://github.com/google-research/robel). May need to download MuJoCo 2.0 if not installed, but it's fine to have multiple versions of MoJoCo.
4. Use Dynamixel Wizard to figure out the port etc of the motors.
5. After modifying ROBEL code, reinstall robel by running

```bash
pip uninstall robel
pip install -e .
```

## Trouble Shooting

If the error `Failed to initialize OpenGL` appears, refer to [link](https://github.com/openai/mujoco-py/issues/187), and try `unset LD_PRELOAD`.

If the error `GLEW initialization error: Missing GL version` appears, refer to [link](https://github.com/openai/mujoco-py/issues/408), and try

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```
