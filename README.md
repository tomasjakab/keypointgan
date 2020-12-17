# [KeypointGAN: Self-supervised Learning of Interpretable Keypoints from Unlabelled Videos](https://www.robots.ox.ac.uk/~vgg/research/unsupervised_pose/)

[Tomas Jakab](http://www.robots.ox.ac.uk/~tomj), Ankush Gupta, Hakan Bilen, Andrea Vedaldi.
CVPR, 2020 (Oral presentation).

## Quick start
Download Simplified Human3.6M dataset from `http://fy.z-yt.net/files.ytzhang.net/lmdis-rep/release-v1/human3.6m/human_images.tar.gz` into `./datasets/simple_human36m/human_images`.

Download a network for perceptual loss from `http://www.robots.ox.ac.uk/~vgg/research/unsupervised_pose/resources/imagenet-vgg-verydeep-19.mat` into `./networks/imagenet-vgg-verydeep-19.mat`.

Paths to datasets and checkpoints can be also customized in `configs/simple_human36m.yaml` and `configs/test_simple_human.yaml`

### Training
Training requires pre-trained keypoint regressor. See bellow for instructions on how to do the pre-training.

A pre-trained regressor can be also downloaded from `http://www.robots.ox.ac.uk/~vgg/research/unsupervised_pose/resources/simple_human36m_regressor/580000_net_regressor.pth`. Save the regressor into `./checkpoints/simple_human36m_regressor` directory unless you specified a different path in the config above.

Train a model on Simplified Human3.6M dataset
```
python2.7 train.py -c configs/simple_human36m.yaml
```

### Testing
Test a model on Simplified Human3.6M dataset
```
python2.7 test_pose.py --test_config configs/test_simple_human.yaml -c configs/simple_human36m.yaml --iteration <ITERATION_NUMBER>
```

## Pre-training regressor
Coming soon.

## Acknowledgments
Parts of the code are based on [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
