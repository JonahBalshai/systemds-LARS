# LARS Optimizer Implementation for ImageNet Training in SystemDS

This directory contains an implementation of the Layer-wise Adaptive Rate Scaling (LARS) optimizer for SystemDS, along with example scripts demonstrating its use with AlexNet on ImageNet.

## Overview

LARS is an optimization algorithm designed for large batch training of neural networks. It was introduced in the paper "Large Batch Training of Convolutional Networks" (You et al., 2017) and later refined in "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes" (You et al., 2019).

### Key Features of LARS:

1. **Layer-wise Learning Rate Adaptation**: LARS computes a local learning rate for each layer based on the ratio of the L2 norm of weights to the L2 norm of gradients.

2. **Large Batch Training**: LARS enables training with very large batch sizes (8K, 16K, 32K) without significant loss in accuracy.

3. **Linear Scaling**: Allows near-linear scaling of learning rate with batch size, crucial for distributed training.

4. **Trust Coefficient**: Uses a trust coefficient (η = 0.001) to control the adaptation strength.

## Implementation

### LARS Optimizer (`scripts/nn/optim/lars.dml`)

The LARS optimizer implements the following update rule:

```
local_lr = η * ||W|| / (||∇L|| + λ||W||)
velocity = μ * velocity + local_lr * global_lr * (∇L + λW)
W = W - velocity
```

Where:
- η: Trust coefficient (default 0.001)
- μ: Momentum coefficient (default 0.9)
- λ: Weight decay coefficient
- ||W||: L2 norm of layer weights
- ||∇L||: L2 norm of layer gradients

### AlexNet Implementation (`scripts/nn/examples/imagenet_alexnet.dml`)

Full AlexNet architecture for ImageNet (224x224 RGB images):
- 5 convolutional layers with ReLU activations
- Batch normalization after each conv layer
- Local Response Normalization (LRN) after conv1 and conv2
- 3 max pooling layers
- 3 fully connected layers with dropout
- 1000-way softmax output

Key features:
- Support for multiple optimizers (SGD, Adam, LARS, etc.)
- Top-1 and Top-5 accuracy metrics
- Batch normalization for training stability
- Proper weight initialization

## Example Scripts

### 1. LARS vs SGD Comparison (`Example-ImageNet_AlexNet_LARS.dml`)

Demonstrates the advantage of LARS for large batch training on ImageNet:

```bash
cd scripts/nn/examples
systemds Example-ImageNet_AlexNet_LARS.dml
```

This script:
- Tests multiple batch sizes: 256, 1024, 4096, 8192
- Compares SGD+Momentum vs LARS
- Shows how LARS maintains performance with large batches
- Measures training time speedup

### 2. Multiple Optimizer Comparison (`Example-ImageNet_AlexNet_Optimizers.dml`)

Comprehensive comparison of different optimizers:

```bash
systemds Example-ImageNet_AlexNet_Optimizers.dml
```

Compares:
- SGD (vanilla)
- SGD with Momentum
- Adam
- LARS

### 3. LARS Unit Test (`test_lars.dml`)

Simple test to verify LARS implementation:

```bash
systemds test_lars.dml
```

## Usage Example

```dml
# Import required modules
source("imagenet_alexnet.dml") as alexnet
source("nn/optim/lars.dml") as lars

# Load ImageNet data (example with dummy data)
X_train = read("imagenet_train.csv", format="csv")
y_train = read("imagenet_train_labels.csv", format="csv")

# ImageNet parameters
C = 3      # RGB channels
H = 224    # Height
W = 224    # Width
K = 1000   # Number of classes

# Training parameters
epochs = 90
batch_size = 8192  # Large batch size
lr = 0.32  # Base learning rate (scaled linearly with batch size)

# Train with LARS
model = alexnet::train(X_train, y_train, X_val, y_val, 
                      C, H, W, epochs, "lars", lr, batch_size)
```

## Hyperparameter Guidelines

### Learning Rate Scaling

For LARS with different batch sizes:
- Batch 256: lr = 0.01 (base)
- Batch 1K: lr = 0.04
- Batch 4K: lr = 0.16
- Batch 8K: lr = 0.32
- Batch 16K: lr = 0.64
- Batch 32K: lr = 1.28

For SGD, use square root scaling:
- lr = base_lr * sqrt(batch_size / 256)

### Other Hyperparameters

**LARS specific:**
- Trust coefficient (η): 0.001
- Momentum (μ): 0.9
- Weight decay (λ): 5e-4

**Training:**
- Epochs: 90
- Learning rate decay: 0.1 at epochs 30, 60, 80
- Data augmentation: Random crop, horizontal flip
- Dropout: 0.5 for FC layers

## Performance Results

Expected results on ImageNet validation set:

| Batch Size | Optimizer | Top-1 Acc | Top-5 Acc | Time/Epoch |
|------------|-----------|-----------|-----------|------------|
| 256        | SGD+Mom   | ~57.5%    | ~80.3%    | Baseline   |
| 8K         | SGD+Mom   | ~53.2%    | ~76.8%    | 0.03x      |
| 8K         | LARS      | ~57.3%    | ~80.1%    | 0.03x      |
| 32K        | LARS      | ~57.1%    | ~79.9%    | 0.008x     |

## Key Insights

1. **Linear Scaling Rule**: LARS allows linear scaling of learning rate with batch size, while SGD requires square root scaling.

2. **Layer-wise Adaptation**: Different layers get different effective learning rates based on their weight/gradient norms, preventing instability.

3. **Bias and BN Parameters**: Small parameters (biases, BN γ and β) use the global learning rate to avoid numerical issues.

4. **Large Batch Efficiency**: LARS is particularly effective for batch sizes > 1K, enabling efficient distributed training.

5. **Warm-up**: For very large batches (>16K), linear warm-up over first 5 epochs is recommended.

## Common Issues and Solutions

### Issue 1: Divergence with Large Batches
**Solution**: Use proper warm-up and ensure trust coefficient is set to 0.001.

### Issue 2: Poor Accuracy with LARS
**Solution**: Check that weight decay is applied correctly (should be included in gradient norm calculation).

### Issue 3: Slow Convergence
**Solution**: Ensure linear learning rate scaling and adjust base learning rate if needed.

## References

1. You, Y., Gitman, I., & Ginsburg, B. (2017). **Large Batch Training of Convolutional Networks**. arXiv:1708.03888

2. You, Y., Li, J., Reddi, S., et al. (2019). **Large Batch Optimization for Deep Learning: Training BERT in 76 minutes**. arXiv:1904.00962

3. Original LARS implementation: https://github.com/NVIDIA/apex

4. Goyal, P., et al. (2017). **Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour**. arXiv:1706.02677

## Future Work

- Implement LAMB (Layer-wise Adaptive Moments optimizer for Batch training)
- Add warm-up scheduling
- Support for mixed precision training
- Integration with distributed training frameworks 