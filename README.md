## Abstract

We study a hierarchical Mixture-of-Experts (MoE) architecture for image classification on TinyImageNet via patch-level Top-1 routing. We analyze the impact of the number of experts (4, 8, 16) on generalization, using specific metrics to monitor both the router's local decision and the global load balancing among experts. The results show that increasing sparse capacity degrades validation performance and is accompanied by traffic imbalance, albeit without causing router collapse or the emergence of dead experts. Finally, we conduct ablation studies to investigate the local capacity of the experts and the impact of dense post-processing, in order to isolate the causes of the observed overfitting.

## Introduction

The use of Mixture-of-Experts (MoE) architectures in computer vision aims to scale model capacity while keeping the active computational cost unchanged. However, the effectiveness of learned routing in hierarchical convolutional structures presents specific challenges, related to specialization on spatial tokens and the stability of load balancing.

In this work, we propose a custom hierarchical Vision MoE architecture tested on TinyImageNet (200 classes, $224 \times 224$ resolution). The main contribution is a systematic analysis of the router's behavior as the expert pool varies. In particular, we distinguish between two different aspects of routing: on one hand, the router's **local decision** on the single token, measured via `spec_entropy`, and on the other hand, the **global balance** of traffic among experts, measured via `entropy_norm_mean`. We show that wider MoE configurations tend to make the router less decisive and simultaneously worsen the load distribution, with a negative effect on Top-1 accuracy. We also investigate whether the observed overfitting is a consequence of excessive local expert expressivity or the interaction between sparse routing and dense residual blocks. The network structure, the Top-1 router, the Fourier positional features, and the aggregated routing metrics are consistent with the project's implementation.

## Method

### Architecture Description

The proposed architecture adopts a ResNet-like structure and adapts it to a hierarchical MoE setting. Each expert is a residual convolutional block of the following type:

$$
\text{Conv}_{3 \times 3} \rightarrow \text{GN} \rightarrow \text{SiLU} \rightarrow \text{Conv}_{3 \times 3} \rightarrow \text{GN},
$$

followed by an internal residual connection within the expert. The MoE layers alternate with dense downsampling blocks. In the code, the experts are implemented as `ConvExpert`, while the transition blocks are implemented as `DownsampleResBlock`.

Each MoE layer contains $E$ experts, a spatial recomposition operation of the processed tokens (`rearrange`), a normalization with GroupNorm, a SiLU, and a shared dense block `post_block`. Denoting the input tokens as $T_{in}$ and the aggregated expert output as $E_{out}$, the layer dynamics can be summarized as:

$$
E_{out} = T_{in} + \alpha \cdot E_{out},
$$

$$
E_{out} = \text{rearrange}(E_{out}),
$$

$$
moe\_out = \text{SiLU}(\text{GN}(E_{out})),
$$

$$
res = post\_block(moe\_out),
$$

$$
moe\_out = moe\_out + res,
$$

$$
X = X + moe\_out.
$$

Here $\alpha$ is a learnable layer parameter, while $X$ represents the feature map passed to the next layer. The shared `post_block` is a dense block of the type:

$post\_block = \text{Conv}_{3 \times 3} \rightarrow \text{GN} \rightarrow \text{SiLU}$

This structure matches the implementation of `PCELayer` and the forward pass of `PCENetwork`.

At each MoE block, the input feature map is decomposed into patches of size $\text{patch\_size} \times \text{patch\_size}$. Proceeding deeper into the network, the `patch_size` value is reduced at downsampling points to a minimum of $2 \times 2$. The patches are also enriched with Fourier positional features prior to routing.

For each patch, the router receives as input the concatenation of the spatial average pooling and max pooling of the enriched token:

$R_{in} = [\text{AVG}(T_{in}); \text{MAX}(T_{in})].$

It then applies LayerNorm and a linear projection to obtain the expert logits:

$l = \text{Linear}(\text{LN}(R_{in})).$

The final probabilities are obtained via:

$S_{exp} = \text{Softmax}(l / \tau)$

and the routing selects the Top-1 expert subject to capacity constraints. This formulation is consistent with `RouterGate` and `Router`.

### Training

Training was performed for 150 epochs with `batch_size = 128` on TinyImageNet. The optimizer used is `AdamW`, with separate parameter groups for the backbone and the router, distinct learning rates, and a warmup + cosine annealing scheduler. In the code, the router is also excluded from weight decay. The training set uses a strong data augmentation pipeline that includes `RandomResizedCrop`, `RandomHorizontalFlip`, `RandomGrayscale`, `ColorJitter`, `RandAugment`, `RandomErasing`, as well as `MixUp` and `CutMix` applied during the training step.

In addition to the task cross-entropy, three auxiliary losses are employed on the router:

* **Load Balancing Loss**:

    $L_{bal} = E \sum_{i=1}^{E} f_i \cdot P_i$

    where $f_i$ is the fraction of tokens assigned to expert $i$ and $P_i$ is the average probability assigned to that expert.

* **Z-Loss**:

    $L_z = \frac{1}{N}\sum_j \left(\log \sum_i e^{l_i(x_j)}\right)^2$

    which penalizes overly large logit magnitudes.

* **Diversity Loss**, which discourages redundant activation patterns among experts by minimizing the correlation between routing probabilities.

## Experiments and Results

### Variation of the Expert Pool

We analyze three MoE configurations (4, 8, 16 experts) against a dense baseline. The table shows that the MoE-4 configuration is the best performing, reaching **63.60%** Top-1 accuracy, a value very close to the dense model's **63.85%**. However, as the number of experts increases, performance progressively degrades.

The `spec_entropy` metric measures the normalized entropy of the router's probability distribution **prior to dispatch**, averaged over the tokens: higher values indicate a less decisive router in the local expert selection. Conversely, `entropy_norm_mean` measures the normalized entropy of the expert utilization distribution **after dispatch**, thus describing how evenly the overall traffic is globally distributed across the pool. In other words, `spec_entropy` captures the local uncertainty of the router, while `entropy_norm_mean` captures global load balancing. This distinction is exactly the one implemented in the `MoEAggregator`.

| Model  | Top-1 Acc ↑ | Val CE ↓ | spec_entropy ↓ |
| ------ | ----------- | -------- | -------------- |
| Dense  | **63.85%** | 1.67     | --             |
| MoE-4  | 63.60%      | **1.66** | 0.50           |
| MoE-8  | 62.06%      | 1.76     | 0.64           |
| MoE-16 | 61.00%      | 1.78     | 0.68           |

For completeness, we also report the average inference cost of the different configurations. In this table, the Top-1 column is calculated on a **dedicated inference dataset**, obtained as a subset of the validation set, while average latency and standard deviation are measured in the same setting. In the considered regime, the introduction of sparse routing does not yield a practical efficiency advantage.

| Model      | Top-1 on inference subset (%) ↑ | Avg Latency (ms) ↓ | Std Latency (ms) ↓ |
| ---------- | ------------------------------- | ------------------ | ------------------ |
| Dense (18) | **67.33** | **2.07** | **0.32** |
| MoE-4      | 65.35                           | 27.61              | 4.47               |
| MoE-8      | 64.36                           | 35.76              | 4.48               |
| MoE-16     | 64.36                           | 47.57              | 4.33               |

### Routing Metrics Analysis

The analysis of dispatching metrics shows that the router does not collapse, but becomes progressively harder to balance as the number of experts increases.

First, `entropy_norm_mean` remains high, signaling that the expert pool remains broadly active. However, this metric alone is not enough to guarantee good balancing. In fact, `imbalance_mean` grows drastically from 9.76 to 90.18 in the 16-expert case, indicating that some experts receive far more traffic than others. Therefore, the issue is not a total routing collapse, but a highly skewed allocation.

Second, the `drop_rate` remains almost constant around 0.02 – 0.03, and the `cap_ratio` remains stable, suggesting that the degradation does not primarily depend on capacity bottlenecks or an excessive number of dropped tokens.

Third, no **dead experts** emerge: in the analyzed runs, the behavior is consistent with $dead\_mean \approx 0$, while `active_mean` remains high. This is important because it indicates that the problem is not the complete inactivity of part of the pool, but rather a severe relative under-utilization of certain experts compared to others. The `dead_mean` and `active_mean` metrics are explicitly tracked in the project's aggregation system.

| Model  | entr_norm | imbalance | drop_rate | cap_ratio |
| ------ | --------- | --------- | --------- | --------- |
| MoE-4  | 0.88      | 9.76      | 0.03      | 0.48      |
| MoE-8  | 0.92      | 24.19     | 0.02      | 0.48      |
| MoE-16 | 0.93      | 90.18     | 0.03      | 0.48      |

## Discussion and Conclusions

A critical observation concerns the trend of the validation loss, which tends to diverge in the final phases of training despite a decreasing training loss. This overfitting, which is more pronounced as the number of experts increases, led us to formulate two architectural hypotheses.

**1. Local capacity of the experts.** In deep layers, where patches reach $2 \times 2$ dimensions, experts with $3 \times 3$ kernels might introduce an artificial receptive field. However, an experimental run in which $3 \times 3$ kernels were replaced with $1 \times 1$ kernels in the last two MoE layers did not reduce overfitting, weakening this hypothesis.

**2. Dense post-processing.** The residual sequence including the dense `post_block` might absorb too much capacity, correcting or overwriting part of the output produced by the sparse experts. To verify this, we modified this component, but observed a worsening in both `validation_class_loss` and Top-1 accuracy. This suggests that dense post-processing remains a useful component for generalization, even though it does not eliminate the overfitting phenomenon. The structure of the `post_block` and the involved layers is consistent with the model's code.

In conclusion, the hierarchical MoE architecture on TinyImageNet shows that greater sparse capacity does not automatically translate into better generalization. The best regime is the one with 4 experts; wider configurations suffer from load imbalance that the auxiliary loss fails to compensate for. The observed overfitting does not seem to depend solely on local kernel sizes or solely on the presence of dense post-processing, but appears primarily linked to the difficulty of stably optimizing traffic allocation among experts.

All results can be reproduced by downloading the entire project repository and running the experimental script `main_experiments.py`, which calls the log analysis and model comparison functions. In the materials you uploaded, this file is present as the dedicated script for final experiments.

***

Would you like me to proofread any specific section of the repository code to ensure the variable names and architectural references match perfectly with this documentation?