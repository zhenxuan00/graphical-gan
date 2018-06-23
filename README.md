# Graphical Generative Adversarial Networks (Graphical-GAN)
## [Chongxuan Li](https://github.com/zhenxuan00), Max Welling, Jun Zhu and Bo Zhang

Code for reproducing most of the results in the [paper](https://arxiv.org/abs/1804.03429). The results of our method is called LOCAL_EP in the code. We also provide implementation of a lot of recent papers, which is of independent interests. The papers including [VEGAN](https://arxiv.org/abs/1705.07642), [ALI](https://arxiv.org/abs/1606.00704), [ALICE](https://arxiv.org/abs/1709.01215). We also try some combination of these methods while the most direct competitor of our method is ALI.

Warning: the code is still under development. If you have any problem with the code, please send an email to chongxuanli1991@gmail.com. Any feedback will be appreciated!

We thank the authors of [wgan-gp](https://github.com/igul222/improved_wgan_training) for providing their code. Our code is widely adapted from their repositories.

You may need to download the datasets and save it to the dataset folder except the MNIST case. See details in the corresponding files of the dataset.

If you find the code is useful, please cite our paper!
``@inproceedings{chen2017population,
  title={Population Matching Discrepancy and Applications in Deep Learning},
  author={Chen, Jianfei and Chongxuan, LI and Ru, Yizhong and Zhu, Jun},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6263--6275},
  year={2017}
}``