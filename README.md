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
<!-- Dijkstra, E. W. (1968). 
Go to statement considered harmful. 
Communications of the ACM, 11(3), 147-148. -->





@article{wan,
  title={Weak adversarial networks for high-dimensional partial differential equations},
  author={Zang, Yaohua and Bao, Gang and Ye, Xiaojing and Zhou, Haomin},
  journal={Journal of Computational Physics},
  volume={411},
  pages={109409},
  year={2020},
  publisher={Elsevier}
}

@article{num,
  title={Numerical solution of inverse problems by weak adversarial networks},
  author={Bao, Gang and Ye, Xiaojing and Zang, Yaohua and Zhou, Haomin},
  journal={Inverse Problems},
  volume={36},
  number={11},
  pages={115003},
  year={2020},
  publisher={IOP Publishing}
}

@article{pinn,
  title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George E},
  journal={Journal of Computational physics},
  volume={378},
  pages={686--707},
  year={2019},
  publisher={Elsevier}
}

@article{drm,
  title={The deep Ritz method: a deep learning-based numerical algorithm for solving variational problems},
  author={Yu, Bing and others},
  journal={Communications in Mathematics and Statistics},
  volume={6},
  number={1},
  pages={1--12},
  year={2018},
  publisher={Springer}
}
