# Implementation of Using WAN(Weak Adversarial Networks) to Solve Inverse Problem

### Files Root directory

* `README.md` explains the whole package.

### Subdirectory

* `model.py` includes the networks, mathematical functions and spacial gradient functions.
* `set_train_params.py` stores the experiment parameters.
* `generate_sample_EIT.py` includes code for generating training and testing samples
* `inv_loss.py` includes loss function for different models.
* `idrm.ipynb, iPINN.ipynb, iwan.ipynb` includes code for execution of the experiment w.r.t. corresponding models.
* `evaluation.py` and `others` are for utility.


## References
<a id="1">[1]</a> 
Zang, Yaohua and Bao, Gang and Ye, Xiaojing and Zhou, Haomin.
Weak adversarial networks for high-dimensional partial differential equations.
Journal of Computational Physics, 411:109409, 2020.

<a id="2">[2]</a> 
Bao, Gang and Ye, Xiaojing and Zang, Yaohua and Zhou, Haomin.
Numerical solution of inverse problems by weak adversarial networks.
Inverse Problems, 36(11):115003, 2020.

<a id='3'>[3]</a> 
Raissi, Maziar and Perdikaris, Paris and Karniadakis, George E.
Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.
Journal of Computational physics, 378:686-707, 2019.

<a id='4'>[4]</a> 
Yu, Bing and others.
The deep Ritz method: a deep learning-based numerical algorithm for solving variational problems.
Communications in Mathematics and Statistics, 6(1):1-12, 2018.
