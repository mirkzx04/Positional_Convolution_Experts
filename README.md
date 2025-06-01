# Project: Positional Convolution Experts

## Abstract

Traditional CNNs process images through a single convolutional channel, potentially limiting the network's ability to capture specific positional information. This project proposes an innovative architecture based on **Positional Convolution Experts** that leverages both content and position of patches to route each patch towards specialized experts, obtaining richer and more representative feature maps.

## Motivation

The main problem of traditional CNNs lies in their inability to effectively exploit positional information of patches within the image. By dividing the image into patches and using both content and position for routing towards specialized experts, each expert can focus on specific characteristics of specific regions of the image.

## Proposed Architecture

### Fundamental Concept

The architecture is based on an intelligent routing system that:
- Divides the image into patches
- Uses content and position information for routing
- Routes each patch towards specialized experts
- Combines results to obtain enriched general feature maps

### Detailed Pipeline

The input image in the network $[B,C,H,W]$ where:
- B → Batch size
- C → Channel (RGB)
- H → Height
- W → Width

The image/feature map is divided into patches of dimensions $hP \times wP$, then applying CoordConv at pixel-level we add spatial information about where pixels are located within the patch and where the patch is located relative to the image, obtaining 
$[B, nP, C +4, H, W]$ where $C+4$ represents the addition of patch coordinates and each pixel within the patches.
The patches are used before training to initialize keys within the router.

The keys are initialized through a $1 \times 1$ convolution on the patch resized as $$[B \times P, C+4, H, W]$$, the result of this convolution is passed to SSP to produce patch embeddings $ Em_p $ which will then be applied to K-Means to obtain centroids that will be used as keys $k \in \mathbb{R^{n_{exp} \times d}}$ where $D = (C+4) \times (1^2 + 2^2 + 4^2)$.

The router will apply cosine similarity between $Em_p$ and $k$ then applying softmax to obtain the selection probabilities of different experts.
The convolutional experts are defined as $Conv_{kz \times kz} - BatchNorm - ReLU$ and will produce different feature maps that will be concatenated through a weighted sum, where the weights will be the probability scores given by the softmax.

Once the global feature map is obtained, it will be re-divided into patches in the manner described above and the patches will be re-applied to the router.

## Training Methodology

### Phase 1: Parameter Stabilization

Three alternative approaches for the initial phase:

#### Option A: Deterministic Routing with Noise
- Position-based patch routing with controlled noise addition
- Experts specialize while maintaining mix caused by noise
- **Advantages**: Guaranteed specialization with diversification

#### Option B: Cosine Similarity with EMA Keys
Uses cosine similarity with keys initialized through SSP for dimensionality standardization while maintaining pixel-level spatial relationships and K-Means to cluster patches around centroids that will then be used as keys to calculate similarity with patch embeddings obtained through SSP.
The patch will be passed through the SSP channel to produce a patch embedding $ Em_p $ that will be used for similarity calculation.
The keys represent moving averages assigned to each patch, maintained outside the computational graph.

#### Option C: Uniform Distribution
Use of a dummy distribution:

$w_i = \frac{\text{number of patches}}{\text{number of experts}}$

### Phase 2: Router Introduction

#### MLP Router
**Characteristics:**
- More computationally heavy
- Requires complete backpropagation
- Pre-processing with CoordConv to enrich patches with positional information

**Pipeline:**
1. Application of CoordConv to the patch
2. Enriched feature map → MLP
3. Output: probability distribution over experts
4. Weighted sum of expert feature maps

#### Key Attention Routing

**Process:**

1. **Key vector formation:**
   $k \in \mathbb{R}^d$

2. **Patch embedding:**
   $p \in \mathbb{R}^{C \times H \times W}$

**Key initialization:**
Dimension standardization with SSP and clustering with K-Means

**Similarity calculation:**

$s_i = \frac{Em_p \cdot k}{||Em_p \cdot k||}$

**Routing weights:**

$w_j = \frac{e^{s_j}}{\sum_j e^{s_j}}$ or softmax

**Final output:**

$\text{out} = \sum_i w_i \cdot E_i(p)$

**Key updates:**
- PyTorch nn.Parameter (Probably applied after a stabilization phase with EMA) to optimize keys through backpropagation
- EMA with backpropagable parameters $k_i^{t+1} = \alpha \cdot k_i^t + (1 - \alpha) \cdot v_i$

#### Gumbel Softmax
Maintains the Option B approach in the pre-training phase with differentiable sampling.

### Phase 3: Attention Introduction

**Pipeline:**
1. Concatenation sum → 1×1 Convolution
2. Resulting feature map → Attention Module
3. Enriched feature map for subsequent routing

**Evaluation:** Performance is monitored to determine module utility.

## Evaluation and Metrics

### Object Detection
**Dataset:** CIFAR-10, Tiny-ImageNet, Pascal VOC

**Metrics:**
- **Accuracy:** Top-1 and Top-5 on CIFAR-10 and Tiny-ImageNet
- **mAP:** Mean Average Precision on Pascal VOC

### Segmentation
**Dataset:** Pascal VOC, Camelyon

**Metrics:**
- **mIoU:** Mean Intersection over Union

## Expected Results

### Performance
- Accuracy improvement compared to traditional CNNs
- Competitive performance with modern CNNs
- Expert specialization based on patch position

### Interpretability
Routing paths offer unique insights:
- **Assignment analysis:** Which patch is assigned to which expert
- **Expert specialization:** Understanding of learned characteristics
- **Architecture optimization:** Removal or reinforcement of experts based on router entropy

### Analysis Metrics
- **Router entropy:** Measure of assignment distribution
- **Route clustering:** Analysis of routing patterns
- **Expert statistics:** Average usage and specialization

## Approach Advantages

1. **Positional Specialization:** Each expert focuses on specific regions
2. **Enriched Feature Maps:** Combination of local and global information
3. **Interpretability:** Ability to analyze routing decisions
4. **Flexibility:** Multiple training and routing approaches
5. **Adaptive Optimization:** Ability to modify the network based on route analysis

## Implementation

The project will be implemented using PyTorch, with particular attention to:
- Code modularity to test different routing options
- Detailed logging for performance analysis
- Routing path visualization for interpretability
- Systematic benchmarking against baseline architectures