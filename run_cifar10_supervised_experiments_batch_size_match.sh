#!/bin/bash
echo "Run supervised experiments"

# Train fully supervised NIN models using subsets of the CIFAR10 training set.
# Use K=5000 training examples per category (which is equal to using the entire training set).
# CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_supervised_NIN_64 # 
# # # Use K=1000 training examples per category.
# CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_supervised_NIN_K1000_64
# # # Use K=400 training examples per category.
# CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_supervised_NIN_K400_64
# # # Use K=100 training examples per category.
# CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_supervised_NIN_K100_64
# # # Use K=20 training examples per category.
# CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_supervised_NIN_K20_64

# CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_supervised_NIN_256 # 
# # # Use K=1000 training examples per category.
# CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_supervised_NIN_K1000_256
# # # Use K=400 training examples per category.
# CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_supervised_NIN_K400_256
# # # Use K=100 training examples per category.
# CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_supervised_NIN_K100_256
# # # Use K=20 training examples per category.
# CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_supervised_NIN_K20_256

# Use K=2500 training examples per category (which is equal to using the entire training set).
CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_supervised_NIN_K2500

CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_supervised_NIN_K2500_64

CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_supervised_NIN_K2500_256
