## Adversarial Inverse Reinforcement Learning with Self-attention Dynamics Model

**[  📺 [Website](https://decisionforce.github.io/MAIRL/) | 🏗 [Github Repo](https://github.com/decisionforce/MAIRL) | 🎓 [Paper](http://bzhou.ie.cuhk.edu.hk/robotics/MAIRL_draft.pdf) ]**

## Dependencies
* Gym >= 0.8.1
* Mujoco-py >= 0.5.7
* Tensorflow >= 1.0.1
* Mujoco

    Add the following path to ~/.bashrc
    ```
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
    export LD_LIBRARY_PATH=$HOME/.mujoco/mjpro150/bin:$LD_LIBRARY_PATH
    ```
    Follow instructions to install [mujoco_py v1.5 here](https://github.com/openai/mujoco-py/tree/498b451a03fb61e5bdfcb6956d8d7c881b1098b5#install-mujoco).

* [SenseAct](https://github.com/kindredresearch/SenseAct) (Optional)

    SenseAct uses Python3 (>=3.5), and all other requirements are automatically installed via pip.
    
    On Linux and Mac OS X, run the following:
    ```
    git clone https://github.com/kindredresearch/SenseAct.git
    cd SenseAct
    pip install -e .
    ```

## How to Run
1. Collect demonstration data and save to `expert_data` directory.

The expert data should be a python pickle file (with `.bin` but not `.pkl` as a suffix) It has `batch_size`, `action`, `states` (required by [set_er_stats()](https://github.com/decisionforce/MAIRL/blob/main/common.py#L19)), like the [`expert_data/hopper_er.bin`](https://github.com/decisionforce/MAIRL/blob/main/expert_data/hopper_er.bin) (just as an example).

2. Training

```bash
COUNTER=1
ENVS+='Ant-v2'
for ENV_ID in ${ENVS[@]}
do
  CUDA_VISIBLE_DEVICES=`expr $COUNTER % 4` python main.py --env_name $ENV_ID --alg mairlImit --obs_mode state &
  COUNTER=$((COUNTER+1))
done
```
3. Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --env_name Ant-v2 --train_mode False
```

# Acknowledgement
Our code is based on [itaicaspi/mgail](https://github.com/itaicaspi/mgail), [HumanCompatibleAI/imitation](https://github.com/HumanCompatibleAI/imitation), [huggingface/transformers](https://github.com/huggingface/transformers).

# Reference
**[Adversarial Inverse Reinforcement Learning with Self-attention Dynamics Model](http://bzhou.ie.cuhk.edu.hk/robotics/MAIRL_draft.pdf)**
<br />
[Jiankai Sun](https://scholar.google.com.hk/citations?user=726MCb8AAAAJ&hl=en),
[Lantao Yu](https://scholar.google.com/citations?user=Ixg9n-EAAAAJ&hl=en), 
[Pinqian Dong](),
[Bo Lu](https://scholar.google.com/citations?user=ENPRTpcAAAAJ&hl=en), and
[Bolei Zhou](https://scholar.google.ca/citations?user=9D4aG8AAAAAJ&hl=en)
<br />
**In IEEE Robotics and Automation Letters (RA-L) 2021**
<br />
[[Paper]](http://bzhou.ie.cuhk.edu.hk/robotics/MAIRL_draft.pdf)
[[Project Page]](https://decisionforce.github.io/MAIRL/)

```
@ARTICLE{sun2021adversarial,
     author={J. {Sun} and L. {Yu} and P. {Dong} and B. {L} and B. {Zhou}},
     journal={IEEE Robotics and Automation Letters},
     title={Adversarial Inverse Reinforcement Learning with Self-attention Dynamics Model},
     year={2021},
}
```
