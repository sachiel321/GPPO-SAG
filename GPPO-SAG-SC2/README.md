# GPPO-SAG

 Guided Proximal Policy Optimization with Structured Action Graph via StarCraft II full game :

-  First version of pre-trained SL and RL agent (only Zerg vs Zerg)
-  Training code of Supervised Learning and Reinforcement Learning of GPPO-SAG

### Installation

Environment requirement:

- Python: 3.10.9

#### 1.Install StarCraftII

- Follow the instruction [here](https://github.com/Blizzard/s2client-proto#downloads)
- Add SC2 installation path to environment variables ``SC2PATH`` :

  - On Linux, input this in terminal:

    ```shell
    export SC2PATH=<sc2/installation/path>
    ```

#### 2.Install requirements:

```bash
pip install -r requirements.txt
```

**Note: GPU is neccessary for decent performance in realtime agent test, you can also use pytorch without cuda, but no performance guaranteed due to inference latency on cpu.
Make sure you set SC2 at lowest picture quality before testing.**

## Train GPPO-SAG with distar pipeline

### Supervised Learning

StarCraftII client is required for replay decoding, follow instructions above.

```bash
python -m distar.bin.sl_train --data <path>
```

*path* could be either a directory with replays or a file includes a replay path at each line.

Optionally, separating replay decoding and model training could be more efficient, run the three scripts in different terminals:

```bash
python -m distar.bin.sl_train --type coordinator
python -m distar.bin.sl_train --type learner --remote
python -m distar.bin.sl_train --type replay_actor --data <path>
```

For distributed training:

```bash
python -m distar.bin.sl_train --init_method <init_method> --rank <rank> --world_size <world_size>
or
python -m distar.bin.sl_train --type coordinator
python -m distar.bin.sl_train --type learner --remote --init_method <init_method> --rank <rank> --world_size <world_size>
python -m distar.bin.sl_train --type replay_actor --data <path>
```

Here is an example of training on a machine with 4 GPUs in remote mode:

```bash
# Run the following scripts in different terminals (windows).
python -m distar.bin.sl_train --type coordinator
# Assume 4 GPUs are on the same machine. 
# If your GPUs are on different machines, you need to configure the init_mehod's IP for each machine.
python -m distar.bin.sl_train --type learner --remote --init_method tcp://127.0.0.1 --rank 0 --world_size 4
python -m distar.bin.sl_train --type learner --remote --init_method tcp://127.0.0.1 --rank 1 --world_size 4
python -m distar.bin.sl_train --type learner --remote --init_method tcp://127.0.0.1 --rank 2 --world_size 4
python -m distar.bin.sl_train --type learner --remote --init_method tcp://127.0.0.1 --rank 3 --world_size 4
python -m distar.bin.sl_train --type replay_actor --data <path>
```

### Reinforcement Learning

Reinforcement learning will use supervised model as initial model, please download it first, StarCraftII client is also required.

##### Training against bots in StarCraftII:

```bash
python -m disatr.bin.rl_train
```

Four components are used for RL training, just like SL training, they can be executed through different process:

```bash
python -m distar.bin.rl_train --type league --task selfplay
python -m distar.bin.rl_train --type coordinator
python -m distar.bin.rl_train --type learner
python -m distar.bin.rl_train --type actor
```

Distributed training is also supported like SL training.



This project is built on an open source AlphaStar implementation:  [DI-star](https://github.com/opendilab/DI-star).

## License

GPPO-SAG released under the Apache 2.0 license.
