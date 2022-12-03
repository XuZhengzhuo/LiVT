# Learning Imbalanced Data with Vision Transformers
Zhengzhuo Xu, Ruikang Liu, Shuo Yang, Zenghao Chai and Chun Yuan

> Abstract: The real-world data tends to be heavily imbalanced and severely skew the data-driven deep neural networks, which
makes Long-Tailed Recognition (LTR) a massive challenging task. Existing LTR methods seldom train Vision Transformers (ViTs) with Long-Tailed (LT) data, while the off-
the-shelf pretrain weight of ViTs always leads to unfair comparisons. In this paper, we systematically investigate the ViTs’ performance in LTR and propose LiVT to train
ViTs from scratch only with LT data. With the observation that ViTs suffer more severe LTR problems, we conduct Masked Generative Pretraining (MGP) to learn gen-
eralized features. With ample and solid evidence, we show that MGP is more robust than supervised manners. In addition, Binary Cross Entropy (BCE) loss, which shows conspicuous performance with ViTs, encounters predicaments in LTR. We further propose the balanced BCE to ameliorate it with strong theoretical groundings. Specially, we
derive the unbiased extension of Sigmoid and compensate extra logit margins to deploy it. Our Bal-BCE contributes
to the quick convergence of ViTs in just a few epochs. Extensive experiments demonstrate that with MGP and Bal-BCE, LiVT successfully trains ViTs well without any additional data and outperforms comparable state-of-the-art methods significantly, e.g., our ViT-B achieves 81.0% Top-1 accuracy in iNaturalist 2018 without bells and whistles.
_________________

This is the PyTorch implementation of [Learning Imbalanced Data with Vision Transformers].

The code will be available soon.
