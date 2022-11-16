# Training Robots to Evaluate Robots: Example-Based Interactive Reward Functions for Policy Learning

CoRL 2022 (Oral)\
[Kun Huang](https://www.linkedin.com/in/kun-huang-620034171/), [Edward S. Hu](https://edwardshu.com/), [Dinesh Jayaraman](https://www.seas.upenn.edu/~dineshj/)
#### [[Paper (Openreview)]](https://openreview.net/forum?id=sK2aWU7X9b8) [[Project Website]](https://sites.google.com/view/lirf-corl-2022/)

<a href="https://sites.google.com/view/lirf-corl-2022/">
<p align="center">
<img src="img/teaser.png" width="600">
</p>
</img></a>

Training visual control policies from scratch on a new robot typically requires generating large amounts of robot-specific data. How might we leverage data previously collected on another robot to reduce or even completely remove this need for robot-specific data? We propose a "robot-aware control" paradigm that achieves this by exploiting readily available knowledge about the robot. We then instantiate this in a robot-aware model-based RL policy by training modular dynamics models that couple a transferable, robot-aware world dynamics module with a robot-specific, potentially analytical, robot dynamics module. This also enables us to set up visual planning costs that separately consider the robot agent and the world. Our experiments on tabletop manipulation tasks with simulated and real robots demonstrate that these plug-in improvements dramatically boost the transferability of visual model-based RL policies, even permitting zero-shot transfer of visual manipulation skills onto new robots. 

If you find this work useful in your research, please cite:

```
@inproceedings{
  hu2022know,
  title={Know Thyself: Transferable Visual Control Policies Through Robot-Awareness},
  author={Edward S. Hu and Kun Huang and Oleh Rybkin and Dinesh Jayaraman},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=o0ehFykKVtr}
}
```




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
