# Positional Convolution Experts (PCE)

## Introduction

This repository implements the **Positional Convolution Experts (PCE)** architecture, an approach to improve the capabilities of traditional convolutional neural networks.

## Motivation

### Limitations of Traditional CNNs

Classic CNNs present a fundamental limitation: they apply the same filter across the entire image, potentially losing region-specific semantic details.

### Practical Example

Consider a complex image containing:
- People wearing different colored clothes
- A road with specific texture
- Trees with detailed foliage  
- Urban panorama (houses, buildings)

Each element has distinct semantics and unique visual characteristics. By applying the same convolutional filter across the entire image, we risk losing:
- The specific color of leaves
- The precise texture of the sidewalk
- Architectural details of buildings
- Chromatic variations in clothing

### PCE Solution

The PCE network solves this problem through:

1. **Specialized Experts**: Each convolutional expert focuses on specific portions of the image
2. **Progressive Resolution**: Patches become smaller and more specific in deeper layers
3. **Intelligent Routing**: A routing system decides which expert is best suited for each patch
4. **Positional Information**: Patches are enriched with precise spatial information

---

## Architecture

### Main Components

The PCE network consists of three fundamental components for each layer:

#### 1. PatchExtractor

**Function**: Division of feature map into enriched patches

**Input**: Feature map $[B, C, H, W]$

**Process**:
- Divides the image into $P$ patches of size $nP \times nP$
- Adds 4 channels of positional information:
  - 2 channels: absolute position of patch in the image
  - 2 channels: relative position of pixels within the patch

**Output**: $[B, P, C+4, H, W]$ where:
- $B$ = batch size
- $P$ = number of patches  
- $C+4$ = original channels + 4 positional channels
- $H$ and $W$ = patch dimensions, height and width respectively

#### 2. Router

**Function**: Routing system to assign patches to experts

**Processing Pipeline**:

1. **Dimensional Projection**:

$$\mathbb{R}^{B \times P \times (C+4) \times N \times N} \xrightarrow{\text{Conv2D}} \mathbb{R}^{B \times P \times 8 \times N \times N}$$

2. **Feature Extraction with SSP**:
   - Applies Spatial Pyramid Pooling with pool sizes $[1, 2, 4]$
   - Produces fixed-dimension patch embeddings

3. **Similarity Calculation**:

$$\text{similarity} = \frac{\text{patch\_embedding} \cdot \text{router\_keys}^T}{||\text{patch\_embedding}|| \cdot ||\text{router\_keys}||}$$

$$\text{weights} = \text{softmax}(\text{similarity})$$

4. **Adaptive Threshold**:
   For adaptive threshold calculation, various statistics are used:

   - $\text{max\_component} = 1 - \text{max\_weight}$ where $\text{max\_weight}$ is the largest weight in $\text{weights}$
   
   - Calculate entropy and maximum entropy: 
      
$$\text{entropy} = -\sum_{i=0}^{n} \text{weights} \cdot \log(\text{weights})$$

$$\text{max\_entropy} = \log(\text{number\_experts})$$

$$\text{norm\_entropy} = \frac{\text{entropy}}{\text{max\_entropy}}$$

$$\text{entropy\_component} = \text{norm\_entropy}$$

   - Calculate Top-1:
     Calculate the probability gap between the most useful expert (highest probability) and all others

$$\text{adaptive\_threshold} = (\text{wth\_max\_c} \cdot \text{max\_component}) + (\text{wth\_entropy\_c} \cdot \text{entropy}) + (\text{wth\_gap\_c} \cdot \text{gap\_component})$$

   Finally, weights are filtered to respect the threshold: 

$$\text{weights\_filtered} = \text{weights} \cdot \text{soft\_mask}$$

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

$$\text{Feature Map} \in \mathbb{R}^{B \times C \times H \times W} \xrightarrow{\text{PatchExtractor}} \mathbb{R}^{B \times P \times (C+4) \times N \times N}$$

2. **Routing Decision**:

$$\mathbb{R}^{B \times P \times (C+4) \times N \times N} \xrightarrow{\text{Router}} \text{Expert Weights} \in \mathbb{R}^{B \times P \times \text{num\_experts}}$$

3. **Expert Processing**:

$$\forall i \in [1, \text{num\_experts}]: \quad \text{patch} \xrightarrow{\text{Expert}_i} \text{feature\_map}_i$$

4. **Weighted Aggregation**:

$$\text{output} = \sum_{i=1}^{\text{num\_experts}} w_i \times \text{feature\_map}_i$$

5. **Reassembly**:

$$\text{Patches} \xrightarrow{\text{Reassemble}} \text{Feature Map} \xrightarrow{\text{Conv1x1}} \text{Output Layer}$$

### Final Output

In the classification layer:

$$\text{Feature Map} \xrightarrow{\text{SPP}} \text{FC Layer} \xrightarrow{} \text{Logits} \in \mathbb{R}^{B \times \text{num\_classes}}$$

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

$$\text{key}_i^{(t+1)} = \alpha \cdot \text{key}_i^{(t)} + (1 - \alpha) \cdot \frac{\sum_j w_{ij} \cdot \text{patch\_embedding}_j}{\sum_j w_{ij}}$$

where $w_{ij}$ is the weight of expert $i$ for patch $j$.

### Phase 2: Fine Tuning (Epochs 150+)

**Transition**:
```python
if epoch >= 150:
    router.set_keys_trainable(True)
```

**Trainable Components**:
- All parameters from Phase 1
- Router keys (full backpropagation)

---

## Technical Features

### Key Initialization

Router keys are initialized through:

1. **K-Means Clustering**:

$$\text{patch\_embeddings} = \text{SSP}(\text{Conv}(\text{patches}))$$

$$\text{centroids} = \text{K-Means}(\text{patch\_embeddings}, k=\text{num\_experts})$$

$$\text{router.keys} = \frac{\text{centroids}}{||\text{centroids}||_2}$$

### Loss Function

**Classification**:

$$\mathcal{L}_{\text{cls}} = \text{CrossEntropy}(\text{logits}, \text{labels})$$

**Router Regularization**:

$$\mathcal{L}_{\text{confidence}} = -\frac{1}{BP} \sum_{b=1}^{B} \sum_{p=1}^{P} \sum_{e=1}^{E} w_{bpe} \log(w_{bpe})$$

$$\mathcal{L}_{\text{collapse}} = \sum_{e=1}^{E} \mathbf{1}_{[\text{usage}_e < 0.01]}$$

where $\text{usage}_e = \frac{1}{BP} \sum_{b,p} \mathbf{1}_{[w_{bpe} > 0]}$

$$\mathcal{L}_{\text{router}} = \alpha \cdot \mathcal{L}_{\text{confidence}} + \beta \cdot \mathcal{L}_{\text{collapse}}$$

**Total Loss**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{router}}$$

---

## Advantages of the Approach

1. **Spatial Specialization**: Each expert learns region-specific features
2. **Computational Efficiency**: Only selected experts process patches
3. **Interpretability**: Routing decisions provide insights into network behavior
4. **Robustness**: Adaptive threshold prevents expert overfitting
5. **Scalability**: Architecture can be extended with more layers and experts