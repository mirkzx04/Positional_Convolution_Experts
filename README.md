# Intro

This repository implements the PCE network.

## Idea

Classic CNNs, even the most modern ones, apply the same filter across the entire image, but this can cause the loss of deep semantic details.
Let's take an image where we have different people with clothes of different colors, a street, trees, and a city panorama (houses, buildings, etc.). Semantically, it's a very rich image. The tree has a well-defined semantic compared to all other elements in the image, and applying the same filter across the entire image could cause the loss of details about the semantics of individual objects, such as the color of leaves or the precise texture of the sidewalk.

PCE attempts to solve this problem by applying convolutional experts to image portions that become progressively smaller as the network depth increases. This means that in deeper layers, experts will specialize in particularly small patches and will therefore be able to learn semantic relationships of very small image portions that will contain very rich information about a specific object, or part of an object, within the entire image.

To decide which expert to delegate the patch to, we use a router that takes as input the patch enriched with two spatial information pieces: the absolute position of the patch within the image and the relative position of pixels within the single patch.

## Network Architecture and Pipeline

### Architecture
Let's dive into the architectural details.

For simplicity, let's analyze a single layer; all other layers repeat in the same structure.

We have 3 main components:
1. **PatchExtractor**: The patch extractor takes the feature map as input and splits it into $ P $ patches of size $ N \times N $, enriching the channel dimension with the absolute position of the patch and relative pixel positions, obtaining a matrix $ [B, P, C+4, H, W] $

2. **Router**: The router takes the matrix $ [B, P, 8, H, W] $. SPP (Spatial Pyramid Pooling) is applied to this matrix to obtain a vector $ [num\_experts, 8] $. This vector is used to find similarity between it and the router keys through CosineSimilarity, to which softmax is applied to obtain expert weights. Each expert has a score (the softmax weight) for each patch. Thanks to an adaptive threshold based on how confident the router feels about its prediction, some experts are discarded (the more confident the router is with its choice, the lower the threshold will be).

    *(N.B. Before the router can take $ [B, P, 8, H, W] $, a convolution is applied to $ [B, P, C+4, H, W] $ to bring the channel to 8)*

3. **Convolutional Expert**: Takes the patch dedicated to it and applies the following operations: *Conv2d -> BatchNorm2d -> ReLU -> Dropout* twice in total (A ResNet block)

### Pipeline

For simplification, let's see the pipeline of a single layer, the operations after applying all experts, and the operations in the output layer.

The layer receives the feature map which is decomposed into patches enriched with positional information that are then passed to the router which will choose which expert (through a weight) to delegate the single patch to.
The patches are passed to the experts that process them through Conv2D, BatchNorm, ReLU, and Dropout (applied twice) and return them as output. Once all expert feature maps are obtained, these are reaggregated through a weighted sum (the weight of feature map $ i $ is that of expert $ i $). This final feature map is reconstructed by reinserting the patches in their appropriate positions, obtaining a new single feature map to which a final convolution is applied.

In the output layer, the incoming feature map is transformed into a vector with SPP which is then passed to an FCL to obtain the final logits for each class in the classification task.

## Training Methodology

The network was trained in 2 phases:
1. **Stabilization**:
In this first phase, the network is trained with backpropagation on expert convolutions, projection convolution, and the final post-expert reassembly convolution, while router keys are updated with EMA.

2. **Fine Tuning**:
Router keys are inserted into the computational graph and therefore backpropagation is also performed on them.

---

# Positional Convolution Experts (PCE)

## Introduction

This repository implements the **Positional Convolution Experts (PCE)** architecture, an approach to improve the capabilities of traditional convolutional neural networks.

## Motivation

### Limitations of Traditional CNNs

Classic CNNs present a fundamental limitation: they apply the same filter across the entire image, potentially losing region-specific semantic details.

### Practical Example

Consider a complex image containing:
- People with clothes of different colors
- A street with specific texture
- Trees with detailed foliage
- Urban panorama (houses, buildings)

Each element has distinct semantics and unique visual characteristics. By applying the same convolutional filter across the entire image, we risk losing:
- The specific color of leaves
- The precise texture of the sidewalk
- Architectural details of buildings
- Chromatic variations in clothing

### PCE Solution

The PCE network solves this problem through:

1. **Specialized Experts**: Each convolutional expert focuses on specific image portions
2. **Progressive Resolution**: Patches become smaller and more specific in deeper layers
3. **Intelligent Routing**: A routing system decides which expert is most suitable for each patch
4. **Positional Information**: Patches are enriched with precise spatial information

---

## Architecture

### Main Components

The PCE network consists of three fundamental components for each layer:

#### 1. PatchExtractor

**Function**: Division of feature map into enriched patches

**Input**: Feature map $[B, C, H, W]$

**Process**:
- Divides the image into $ P $ patches of size $ nP \times nP $
- Adds 4 channels of positional information:
  - 2 channels: absolute position of patch in image
  - 2 channels: relative position of pixels in patch

**Output**: $[B, P, C+4, H, W]$ where:
- $B$ = batch size
- $P$ = number of patches
- $C+4$ = original channels + 4 positional channels
- $H$ and $ W $ = patch dimensions, height and width respectively

#### 2. Router

**Function**: Routing system to assign patches to experts

**Processing Pipeline**:

1. **Dimensional Projection**:
   $\mathbb{R}^{B \times P \times (C+4) \times N \times N} \xrightarrow{\text{Conv2D}} \mathbb{R}^{B \times P \times 8 \times N \times N}$

2. **Feature Extraction with SPP**:
   - Applies Spatial Pyramid Pooling with pool sizes $[1, 2, 4]$
   - Produces fixed-dimension patch embedding

3. **Similarity Calculation**:
   $\text{similarity} = \frac{\text{patch\_embedding} \cdot \text{router\_keys}^T}{||\text{patch\_embedding}|| \cdot ||\text{router\_keys}||}$
   $\text{weights} = \text{softmax}(\text{similarity})$

4. **Adaptive Threshold**:
   For adaptive threshold calculation, various statistics are used:

   - $max\_component = 1 - max\_weight$ where $ max\_weight $ is the largest weight in $ weights $
   - We calculate entropy and maximum entropy:
      
      $ entropy = \sum_{i=0}^{n} weights \cdot \log(weights) $, $ max\_entropy = \log(number\_experts) \implies norm\_entropy = \frac{entropy}{max\_entropy}$
   
      $ entropy\_component = norm\_entropy $

   - We calculate Top-1:
      We calculate the probability gap between the most useful expert (the one with highest probability) and all others

   $ adaptive\_threshold = (wth\_max\_c \cdot max\_component) + (wth\_entropy\_c \cdot entropy) + (wth\_gap\_c \cdot gap\_component) $

   Finally, weights are filtered to respect the threshold: $ weights\_filtered = weights \cdot soft\_mask $

**Output**: Normalized weights for each expert $\mathbf{W} \in \mathbb{R}^{B \times P \times E}$ where $E$ = number of experts

#### 3. Convolutional Experts

**Architecture**: ResNet-like block

**Structure**:
```
Conv2D → BatchNorm → ReLU → Dropout
    ↓
Conv2D → BatchNorm → (+) → ReLU
                      ↑
                 Skip Connection
```

**Function**: Specialized processing of assigned patches

---

## Complete Pipeline

### Single Layer Flow

1. **Patch Extraction**:
   $\text{Feature Map} \in \mathbb{R}^{B \times C \times H \times W} \xrightarrow{\text{PatchExtractor}} \mathbb{R}^{B \times P \times (C+4) \times N \times N}$

2. **Decision Routing**:
   $\mathbb{R}^{B \times P \times (C+4) \times N \times N} \xrightarrow{\text{Router}} \text{Expert Weights} \in \mathbb{R}^{B \times P \times \text{num\_experts}}$

3. **Expert Processing**:
   $\forall i \in [1, \text{num\_experts}]: \quad \text{patch} \xrightarrow{\text{Expert}_i} \text{feature\_map}_i$

4. **Weighted Aggregation**:
   $\text{output} = \sum_{i=1}^{\text{num\_experts}} w_i \times \text{feature\_map}_i$

5. **Reassembly**:
   $\text{Patches} \xrightarrow{\text{Reassemble}} \text{Feature Map} \xrightarrow{\text{Conv1x1}} \text{Output Layer}$

### Final Output

In the classification layer:
$\text{Feature Map} \xrightarrow{\text{SPP}} \text{FC Layer} \xrightarrow{} \text{Logits} \in \mathbb{R}^{B \times \text{num\_classes}}$

---

## Training Methodology

Training occurs in **two distinct phases**:

### Phase 1: Stabilization (Epochs 0-150)

**Trainable Components**:
- Expert convolutions
- Projection convolutions
- Final post-reassembly convolutions

**Fixed Components**:
- Router keys (updated via EMA)

**EMA Update**:
$\text{key}_i^{(t+1)} = \alpha \cdot \text{key}_i^{(t)} + (1 - \alpha) \cdot \frac{\sum_j w_{ij} \cdot \text{patch\_embedding}_j}{\sum_j w_{ij}}$

where $w_{ij}$ is the weight of expert $i$ for patch $j$.

### Phase 2: Fine Tuning (Epochs 150+)

**Transition**:
```python
if epoch >= 150:
    router.set_keys_trainable(True)
```

**Trainable Components**:
- All parameters from Phase 1
- Router keys (complete backpropagation)

---

## Technical Features

### Key Initialization

Router keys are initialized through:

1. **K-Means Clustering**:
   $\text{patch\_embeddings} = \text{SPP}(\text{Conv}(\text{patches}))$
   $\text{centroids} = \text{K-Means}(\text{patch\_embeddings}, k=\text{num\_experts})$
   $\text{router.keys} = \frac{\text{centroids}}{||\text{centroids}||_2}$

### Loss Function

**Classification**:
$\mathcal{L}_{\text{cls}} = \text{CrossEntropy}(\text{logits}, \text{labels})$

**Router Regularization**:
$\mathcal{L}_{\text{confidence}} = -\frac{1}{BP} \sum_{b=1}^{B} \sum_{p=1}^{P} \sum_{e=1}^{E} w_{bpe} \log(w_{bpe})$

$\mathcal{L}_{\text{collapse}} = \sum_{e=1}^{E} \mathbf{1}_{[\text{usage}_e < 0.01]}$

where $\text{usage}_e = \frac{1}{BP} \sum_{b,p} \mathbf{1}_{[w_{bpe} > 0]}$

$\mathcal{L}_{\text{router}} = \alpha \cdot \mathcal{L}_{\text{confidence}} + \beta \cdot \mathcal{L}_{\text{collapse}}$

**Total Loss**:
$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{router}}$

---

## Advantages of the Approach

1. **Spatial Specialization**: Each expert learns region-specific features
2. **Computational Efficiency**: Only selected experts process patches
3. **Interpretability**: Routing decisions provide insights into network behavior
4. **Robustness**: Adaptive threshold prevents expert overfitting
5. **Scalability**: Architecture can be extended with more layers and experts