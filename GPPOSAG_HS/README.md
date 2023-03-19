# GPPO-SAG

*uided Proximal Policy Optimization with Structured Action Graph via Hearthstone:*

This project is tested with `python==3.9.5` and `Ubuntu 18.04`.

## Requirements

Python Packages:

```
pip install -r requirements.txt
```

Others:

- Install [Mono](https://www.mono-project.com) to support .NET Framework on Ubuntu

## Training

To train the Policy Model, run:

```
python main.py
```

## Evaluation

You can use ``winrate_compute.py`` or `winrate_compute_multi.py` to record a game log and see how the agent performs.
