# Positional Convolution Experts (PCE)

### Introduction

This repository implements the **Positional Convolution Experts (PCE)** network architecture.

### Problem

Traditional CNNs, even modern ones, apply the same filter across the entire image, which can cause the loss of deep semantic details. Consider an image with different people wearing various colored clothes, a street, trees, and an urban panorama (houses, buildings, etc.). Semantically, this is a rich image where each element has well-defined semantics. Applying the same filter across the entire image could lose details about the semantics of individual objects, such as leaf colors or precise sidewalk texture.

### Solution

PCE addresses this problem by applying convolutional experts to image portions that become progressively smaller as network depth increases. In deeper layers, experts specialize on particularly small patches, learning semantic relationships of tiny image portions containing rich information about specific objects or object parts within the entire image.

To decide which expert to delegate each patch to, we use a router that takes as input the patch enriched with two spatial information types: the absolute position of the patch within the image and the relative position of pixels within the individual patch.

### Architecture

#### Main Components

1. **PatchExtractor**: Takes the feature map as input and splits it into P patches of size N×N, enriching the channel dimension with absolute patch position and relative pixel position, obtaining a matrix [B, P, C+4, H, W]

2. **Router**: Takes the matrix [B, P, 8, H, W] and applies SPP (Spatial Pyramid Pooling) to obtain a vector. This vector finds similarity with router keys through CosineSimilarity, then applies softmax to get expert weights. Each expert has a score for each patch. Thanks to an adaptive threshold based on router confidence, some experts are discarded (higher router confidence = lower threshold)

3. **Convolutional Expert**: Takes its dedicated patch and applies: Conv2d → BatchNorm2d → ReLU → Dropout (applied twice, forming a ResNet block)

#### Pipeline

The layer receives the feature map, which is decomposed into patches enriched with positional information. These are passed to the router, which chooses which expert (through a weight) to delegate each patch to. Patches are passed to experts who process them through Conv2D, BatchNorm, ReLU, and Dropout (applied twice) and return them as output. Once all expert feature maps are obtained, they are reaggregated through weighted sum (the weight of feature map i is that of expert i). This final feature map is reconstructed by reinserting patches into their appropriate positions, obtaining a new single feature map to which a final convolution is applied.

In the output layer, the input feature map is transformed into a vector with SPP, then passed to an FCL to obtain final logits for each class in the classification task.

### Training Methodology

The network is trained in 2 phases:

1. **Stabilization**: The network is trained with backpropagation on expert convolutions, projection convolution, and final post-reassembly convolution, while router keys are updated with EMA

2. **Fine Tuning**: Router keys are inserted into the computational graph, so backpropagation is also performed on them

### Usage

```python
from Model.PCE import PCENetwork
from Model.Components.Router import Router

# Initialize router and model
router = Router(num_experts=4, temperature=1.0)
model = PCENetwork(
    input_channels=7,  # 3 RGB + 4 positional
    num_experts=4,
    patch_size=16,
    router=router,
    num_classes=10
)

# Training setup
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = PCEScheduler(optimizer, phase_epochs=[150, 100])
```