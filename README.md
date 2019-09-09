# human-motion-prediction-pytorch
This is the Pytorch implementation for the paper

Julieta Martinez, Michael J. Black, Javier Romero.
_On human motion prediction using recurrent neural networks_. In CVPR 17.

It can be found on arxiv as well: https://arxiv.org/pdf/1705.02445.pdf

### Dependencies

* [h5py](https://github.com/h5py/h5py) -- to save samples
* Pytorch: the code is tested on 1.0.0 with CUDA 10.0.0.

### Get this code and the data

First things first, clone this repo and get the human3.6m dataset on exponential map format.

```bash
git clone https://github.com/una-dinosauria/human-motion-prediction.git
cd human-motion-prediction
mkdir data
cd data
wget http://www.cs.stanford.edu/people/ashesh/h3.6m.zip
unzip h3.6m.zip
rm h3.6m.zip
cd ..
```

### Quick demo and visualization

For a quick demo, you can train for a few iterations and visualize the outputs
of your model.

A simple demonstration is available by running
```bash
bash demo.sh
```
For further training and testing, you can change the command with following instruction:

To train,
```bash
python src/translate.py --action walking --seq_length_out 25 --iterations 10000
```

To save some samples of the model,
```bash
python src/translate.py --action walking --seq_length_out 25 --iterations 10000 --sample --load 10000
```

Finally, to visualize the samples,
```bash
python src/forward_kinematics.py
```
If it works, it should produce some visualization like

![Walking](https://raw.githubusercontent.com/garroud/human-motion-prediction-pytorch/master/figs/walking_py.gif)

### RNN models

To train and reproduce the results of our models, use the following commands

| model      | arguments | training time (gtx 1080) | notes |
| ---        | ---       | ---   | --- |
| Sampling-based loss (SA) | `python src/translate.py --action walking --seq_length_out 25` | 45s / 1000 iters | Realistic long-term motion, loss computed over 1 second. |
| Residual (SA)            | `python src/translate.py --residual_velocities --action walking` | 35s / 1000 iters |  |
| Residual unsup. (MA)     | `python src/translate.py --residual_velocities --learning_rate 0.005 --omit_one_hot` | 65s / 1000 iters | |
| Residual sup. (MA)       | `python src/translate.py --residual_velocities --learning_rate 0.005` | 65s / 1000 iters | best quantitative.|
| Untied       | `python src/translate.py --residual_velocities --learning_rate 0.005 --architecture basic` | 70s / 1000 iters | |


You can substitute the `--action walking` parameter for any action in

```
["directions", "discussion", "eating", "greeting", "phoning",
 "posing", "purchases", "sitting", "sittingdown", "smoking",
 "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]
```

or `--action all` (default) to train on all actions.

### Result visualization
Here is some result visualization on current models.

<img src="https://raw.githubusercontent.com/garroud/human-motion-prediction-pytorch/irl_training/figs/walking_py.gif" width=350> <img src="https://raw.githubusercontent.com/garroud/human-motion-prediction-pytorch/irl_training/figs/eating_py.gif" width=350>
<img src="https://raw.githubusercontent.com/garroud/human-motion-prediction-pytorch/irl_training/figs/greeting_py.gif" width=350>
<img src="https://raw.githubusercontent.com/garroud/human-motion-prediction-pytorch/irl_training/figs/phoning_py.gif" width=350>
<img src="https://raw.githubusercontent.com/garroud/human-motion-prediction-pytorch/irl_training/figs/posing_py.gif" width=350>
<img src="https://raw.githubusercontent.com/garroud/human-motion-prediction-pytorch/irl_training/figs/purchases_py.gif" width=350>
<img src="https://raw.githubusercontent.com/garroud/human-motion-prediction-pytorch/irl_training/figs/sittingdown_py.gif" width=350>
<img src="https://raw.githubusercontent.com/garroud/human-motion-prediction-pytorch/irl_training/figs/sitting_py.gif" width=350>
<img src="https://raw.githubusercontent.com/garroud/human-motion-prediction-pytorch/irl_training/figs/smoking_py.gif" width=350>
<img src="https://raw.githubusercontent.com/garroud/human-motion-prediction-pytorch/irl_training/figs/takingphoto_py.gif" width=350>
<img src="https://raw.githubusercontent.com/garroud/human-motion-prediction-pytorch/irl_training/figs/waiting_py.gif" width=350>
<img src="https://raw.githubusercontent.com/garroud/human-motion-prediction-pytorch/irl_training/figs/walkingdog_py.gif" width=350>
<img src="https://raw.githubusercontent.com/garroud/human-motion-prediction-pytorch/irl_training/figs/walking_py.gif" width=350>
<img src="https://raw.githubusercontent.com/garroud/human-motion-prediction-pytorch/irl_training/figs/walkingtogether_py.gif" width=350>
### Citing

If you use our code, please cite our work

```
@inproceedings{julieta2017motion,
  title={On human motion prediction using recurrent neural networks},
  author={Martinez, Julieta and Black, Michael J. and Romero, Javier},
  booktitle={CVPR},
  year={2017}
}
```

### Acknowledgments

The main part of the program is based on the original [TensorFlow implementation](https://github.com/una-dinosauria/human-motion-prediction).

The pre-processed human 3.6m dataset and some of our evaluation code (specially under `src/data_utils.py`) was ported/adapted from [SRNN](https://github.com/asheshjain399/RNNexp/tree/srnn/structural_rnn) by [@asheshjain399](https://github.com/asheshjain399).
### Licence
MIT
