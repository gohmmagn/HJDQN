Hamilton-Jacobi DQN for Infinite Horizon Optimal Control Problems governed by parabolic PDEs
====================================================

This repository includes an official PyTorch implementation of **Hamilton-Jacobi DQN**
and [DDPG][ddpglink] as a baseline. The original framework was introduced by **[HJDQN framework][hjdqnlink]**.
This project is used in the Google Colab environment via the HJDQN jupyter notebook. 

## 1. Requirements

The followings packages must be installed for either local or colab setup:

- **[Gymnasium][gymlink]**

- **[Pytorch][pytorchlink]**

- **[FEniCSx][fenicsxlink]**

- **[Python Control System Library][controllink]**

## 2. Installation

For colab setup just follow the instructions in the HJDQ jupyter notebook.

[controllink]: https://python-control.readthedocs.io/en/0.9.4/
[fenicsxlink]: https://fenicsproject.org/
[hjdqnlink]: https://arxiv.org/abs/2010.14087
[ddpglink]: https://arxiv.org/abs/1509.02971
[gymlink]: https://gymnasium.farama.org/
[pytorchlink]: https://pytorch.org/
