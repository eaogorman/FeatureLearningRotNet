!/bin/bash

# Train a RotNet (with a NIN architecture of 4 conv. blocks) on training images of CIFAR10.
CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_RotNet_NIN4blocks_64

CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_RotNet_NIN4blocks_256