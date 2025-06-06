# ResNet with LARS Optimizer for ImageNet

## Overview

This implementation provides ResNet-18 training on ImageNet with support for the LARS (Layer-wise Adaptive Rate Scaling) optimizer. LARS enables efficient large batch training by adapting learning rates per layer based on the ratio of weight norm to gradient norm.

## Key Features

- **ResNet-18 Architecture**: Leverages existing SystemDS ResNet-18 implementation
- **LARS Optimizer**: Layer-wise adaptive learning rates for large batch training
- **Multiple Optimizer Support**: Compare LARS with SGD, SGD+Momentum, and Adam
- **ImageNet Scale**: Designed for 224x224 RGB images with 1000 classes
- **Top-1 and Top-5 Accuracy**: Standard ImageNet evaluation metrics

## Files

### Core Implementation
- `imagenet_resnet.dml`: ResNet-18 wrapper with LARS support
  - Integrates with existing ResNet-18 architecture
  - Adds LARS optimizer update function
  - Provides training and evaluation functions

### Example Scripts
- `Example-ImageNet_ResNet_LARS.dml`: LARS vs SGD comparison
  - Tests multiple batch sizes (256, 1K, 4K, 8K)
  - Demonstrates linear vs sqrt learning rate scaling
  
- `Example-ImageNet_ResNet_Optimizers.dml`: Comprehensive optimizer comparison
  - Compares SGD, SGD+Momentum, Adam, and LARS
  - Fixed batch size comparison

## LARS Algorithm

The LARS optimizer computes layer-wise learning rates:

```
local_lr = trust_coeff * ||W|| / (||∇L + λW|| + ε)
velocity = momentum * velocity + local_lr * global_lr * (∇L + λW)
W = W - velocity
```

Where:
- `||W||` is the L2 norm of weights
- `||∇L + λW||` is the L2 norm of gradients with weight decay
- `trust_coeff` controls the adaptation strength (default: 0.001)
- `ε` prevents division by zero (default: 1e-9)

## Usage

### Basic Training

```dml
source("imagenet_resnet.dml") as resnet

# Load ImageNet data (not shown)
X_train = ...  # Shape: (N, 3*224*224)
y_train = ...  # Shape: (N, 1000) one-hot encoded

# Train with LARS
[model, emas] = resnet::train(X_train, y_train, X_val, y_val,
                             C=3, H=224, W=224, epochs=90,
                             optimizer="lars", lr=0.1, batch_size=256)
```

### Large Batch Training

```dml
# LARS enables linear learning rate scaling
batch_size = 8192
lr = 0.1 * (batch_size / 256)  # Linear scaling

[model, emas] = resnet::train(X_train, y_train, X_val, y_val,
                             C=3, H=224, W=224, epochs=90,
                             optimizer="lars", lr=lr, batch_size=batch_size)
```

### Evaluation

```dml
# Evaluate trained model
[loss, top1_acc, top5_acc] = resnet::evaluate(X_test, y_test, 
                                              C=3, H=224, W=224,
                                              model, emas)

print("Test Loss: " + loss)
print("Top-1 Accuracy: " + top1_acc + "%")
print("Top-5 Accuracy: " + top5_acc + "%")
```

## Hyperparameters

### LARS Specific
- **Trust Coefficient (η)**: 0.001 (controls adaptation strength)
- **Weight Decay (λ)**: 5e-4
- **Momentum (μ)**: 0.9
- **Base Learning Rate**: Scale linearly with batch size

### Learning Rate Scaling
- **LARS**: `lr = base_lr * (batch_size / 256)` (linear)
- **SGD**: `lr = base_lr * sqrt(batch_size / 256)` (square root)

### Recommended Settings
| Batch Size | LARS LR | SGD LR | Epochs |
|------------|---------|---------|---------|
| 256        | 0.1     | 0.1     | 90      |
| 1024       | 0.4     | 0.2     | 90      |
| 4096       | 1.6     | 0.4     | 90      |
| 8192       | 3.2     | 0.57    | 90      |

## Performance Expectations

### Accuracy (90 epochs)
- **Small Batch (256)**: ~76% top-1, ~93% top-5
- **Large Batch (8K) with LARS**: ~75% top-1, ~92% top-5
- **Large Batch (8K) with SGD**: ~73% top-1, ~91% top-5

### Training Speed
- Larger batches = fewer iterations = faster training
- 8K batch: ~8x fewer iterations than 1K batch
- Actual speedup depends on hardware utilization

## Implementation Details

### Integration with ResNet-18
The implementation leverages the existing ResNet-18 architecture in SystemDS:
- Uses `resnet18::forward()` for forward pass
- Uses `resnet18::backward()` for gradient computation
- Adds custom `update_params_with_lars()` function

### Parameter Update
LARS updates are applied to all parameters including:
- Convolutional weights
- Batch normalization scales and biases
- Fully connected layer weights

### Small Parameter Handling
Parameters with small norms (e.g., biases) use simplified update:
- If `||W|| < 1e-3`, skip adaptive scaling
- Prevents numerical instability

## Comparison with Other Optimizers

### SGD
- Requires careful learning rate tuning
- Performance degrades with large batches
- Square root scaling rule

### SGD with Momentum
- Better than vanilla SGD
- Still suffers from large batch degradation
- Popular baseline for vision tasks

### Adam
- Adaptive per-parameter learning rates
- Works well for moderate batch sizes
- May not scale to very large batches

### LARS
- Designed specifically for large batch training
- Layer-wise adaptation prevents gradient explosion
- Enables linear learning rate scaling

## Tips for Best Results

1. **Warm-up**: Consider learning rate warm-up for very large batches
2. **Batch Size**: LARS shines with batch sizes ≥ 1024
3. **Learning Rate**: Use linear scaling rule with LARS
4. **Weight Decay**: Keep at 5e-4 for ImageNet
5. **Momentum**: 0.9 works well in most cases

## References

1. You et al. (2017): "Large Batch Training of Convolutional Networks"
   - Original LARS paper
   - https://arxiv.org/abs/1708.03888

2. He et al. (2016): "Deep Residual Learning for Image Recognition"
   - ResNet architecture
   - https://arxiv.org/abs/1512.03385

3. Goyal et al. (2017): "Accurate, Large Minibatch SGD"
   - Large batch training techniques
   - https://arxiv.org/abs/1706.02677

## Future Enhancements

1. **More ResNet Variants**: Add ResNet-34, 50, 101, 152
2. **LAMB Optimizer**: Layer-wise Adaptive Moments (LARS + Adam)
3. **Learning Rate Schedules**: Cosine annealing, polynomial decay
4. **Mixed Precision**: FP16 training support
5. **Distributed Training**: Multi-GPU/node support 