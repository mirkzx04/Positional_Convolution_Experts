## Abstract

This project aims to study the training dynamics of a Vision Mixture-of-Experts (MoE) architecture in a context different from classic Transformer architectures.

Specifically, the project analyzes an MoE embedded within a hierarchical CNN. Unlike Vision Transformers, where tokens can interact globally via attention, a hierarchical CNN operates with a local and progressively downsampled structure. This makes MoE routing on patches an interesting case: the router must specialize the experts using local and positional information, without directly benefiting from a global attention mechanism.

Experiments suggest that, with the current setup, a CNN-based MoE may suffer from a load imbalance in the assignments made to the experts by the router. This imbalance appears to be correlated with a degradation in generalization, but the cause of the overfitting is not completely isolated. In fact, tests show that dense residual blocks also have a strong impact on performance, leaving the relationship between sparse routing, dense post-processing, imbalance, and overfitting an open question.

---

## Introduction

In this work, an MoE architecture based on a hierarchical CNN is proposed. The convolutional component resides in the model's backbone, while sparse routing is applied at the patch level on intermediate feature maps.

The model was trained on TinyImageNet, which consists of 200 classes and images resized to 224 × 224. The main contribution of this project is the analysis of the routing dynamics. In particular, two aspects are distinguished:

1. **Local router decision**, measured via `spec_entropy`, which indicates how sharp or uniform the probability distribution over the experts is.
2. **Global traffic balancing**, measured via `entropy_norm_mean` and `imbalance_mean`, which describe how the patches are distributed among the experts.

## Method

### Architecture 
<img src="../img/PCE_architecture.png" width="750" align="left" style="margin-right: 20px; margin-bottom: 10px;">

**MoE Blocks** The MoE blocks operate at the patch level on the feature map. Each input feature map is divided by the `Patch Extractor` into patches of size $P_s \times P_s$. For each patch, positional features are then computed using `Fourier Positional Features`, which are concatenated to the patch content and used exclusively by the router.

The router therefore receives patches enriched with positional information and decides, via Top-1 routing, which expert to assign each patch to. The experts, on the other hand, receive only the original patch content, without positional embeddings. In this way, spatial specialization is delegated to the router: it is the router that, observing both the content and the position of the patch in the original feature map, must select the most suitable expert.

After routing, each convolutional expert processes only the patches assigned to it. The experts' outputs are then weighted by the combination coefficient produced by the router and accumulated in the position corresponding to the original patch.

Denoting the aggregated output of the experts as $E_{out}$ and the original input patches to the block as $X_p$, the first residual is defined as:
$$O = X_p + \gamma E_{out}$$
where $\gamma$ is a learnable parameter initialized to a small value. This residual serves two main functions: preserving the original signal during the early stages of training and making the experts' contribution gradual, preventing the sparse routing from destabilizing the representation too early.

At this point, the processed patches are recomposed into their original spatial arrangement via `rearrange`:
$$R_{out}(O) \in \mathbb{R}^{B \times C_{out} \times H_{out} \times W_{out}}$$

The reconstructed feature map is then normalized and activated:
$$MoE_{out} = \text{SiLU}(\text{GN}(R_{out}(O)))$$
This step serves to stabilize the feature distribution after the sparse processing and before the subsequent dense phase.

The third block is a shared convolutional block:
$$res = \text{Conv}_{3 \times 3} \rightarrow \text{GN} \rightarrow \text{SiLU}$$
Its role is to locally mix the features produced by the experts. This is important because the patches are processed separately and then reinserted into their original positions: without a local dense operation, discontinuities could emerge between adjacent patches or overly sharp boundaries between regions processed by different experts.

Finally, the layer applies two residual connections:
$$MoE_{out} = MoE_{out} + res$$
$$X_{out} = R_{out}(O) + MoE_{out}$$
The first residual integrates the contribution of the dense convolutional block, while the second maintains a direct connection with the reconstructed feature map after the experts' processing. In this way, the block combines three components: sparse routing on the experts, shared normalization/activation, and dense convolutional post-processing.

**Router Gate** The `Router Gate` is the module that assigns each patch to one of the available experts. For each patch $X_p$, the gate constructs a compact representation using global spatial statistics:
$$R_{in} = [\text{mean}(X_p); \text{amax}(X_p)]$$
where $\text{mean}(X_p)$ and $\text{amax}(X_p)$ are calculated along the spatial dimensions of the patch. In this way, each patch is represented by a vector that summarizes both its mean activation and its maximum activation.

This representation is normalized and passed to a linear layer:
$$l = W \cdot \text{LN}(R_{in}) + b$$
where $l \in \mathbb{R}^{N_{exp}}$. The vector $l$ contains a logit for each expert. The `Router Gate` can therefore be seen as a function that, given the representation of a patch, assigns a score to each expert in the pool.

The logits are then scaled by a temperature $\tau$ and transformed into probabilities via softmax: 
$$p_{exp} = \text{softmax}\left(\frac{l}{\tau}\right)$$

The temperature controls how uniform or selective the distribution over the experts is: higher values of $\tau$ produce softer distributions, while lower values make the logits sharper and push the router toward more distinct choices.

Once the distribution over the experts is obtained, a `Top-1` routing is applied: for each patch, the expert with the highest probability is selected.
$$e^* = \arg\max_i \ p_i$$

However, each expert has a maximum capacity, denoted as `capacity`, which limits the number of patches it can process in a single forward pass. The capacity is calculated based on the total number of patches, the number of experts, and the `capacity_factor`:
$$C_{cap} = \left\lceil \frac{\text{capacity\_factor} \cdot N}{N_{exp}} \right\rceil$$
where $N$ is the total number of patches/tokens to be routed. If an expert receives more patches than its capacity, only the patches with the highest routing probability are retained, while the others are dropped from the sparse path.

The dropped patches are not processed by the experts; however, thanks to the residual connection of the MoE block, the original signal of the patch can still be preserved in the layer. The capacity thus serves to control the computational load of the experts and to prevent a single expert from absorbing all the traffic.

<br clear="all">

---

### Training 
### Training 
The model was trained on TinyImageNet with 224 × 224 images. The following setup was used for all training sessions: 

| Learning Rate Backbone | Learning Rate Router | Batch Size | Stem Kernel | Stem Out Channels | Optimizer | Weight Decay |
|------------------------|----------------------|------------|-------------|-------------------|-----------|--------------|
| 0.001                  | 0.001                | 128        | 7x7         | 64                | AdamW     | 1e-3         |

In the setup, the Backbone and Router share the optimizer but not the learning rate. The choice of a higher learning rate for the router stems from the fact that it must adapt quickly to the features extracted by the backbone. Since these features are very rich, thanks to the experts' computation and enrichment via residual connections, the risk of setting a low learning rate for the router is that it might fail to adapt quickly. Furthermore, the router is exempt from weight decay.

Both the backbone and router learning rates go through a warmup phase: 
1. **Backbone Warmup**: Occurs during the first training epochs.
2. **Router Warmup**: In the early training epochs, the router assigns patches uniformly to all experts until each one's capacity is filled. During this phase, the router is completely frozen and thus receives no backpropagation; after this phase, the router goes through its warmup phase.

| Tau Init | Tau Final | Aux Loss Weight Init | Aux Loss Final |
|----------|-----------|----------------------|----------------|
| 2.0      | 0.85      | 0.05                 | 5e-4           |

The router is also equipped with additional hyperparameters: Tau serves to regulate the temperature in the logits, and Aux Loss Weight serves to regulate the weight of the load balancing loss.

$$
\alpha =
\begin{cases}
0 & e < e_{router} \\
\alpha_{peak} \cdot \frac{e - e_{router} + 1}{e_{warmup}} & e_{router} \le e < e_{router} + e_{warmup} \\
\alpha_{peak} & e_{router} + e_{warmup} \le e < e_{decay} \\
\alpha_{final} + (\alpha_{peak} - \alpha_{final}) \cdot \frac{1 + \cos(\pi t)}{2} & e \ge e_{decay}
\end{cases}
$$

In addition to these, there is also noise, which injects noise into the logits for exploration purposes in the router.
All these hyperparameters are scheduled during training; the noise reaches 0.0 at the end of training.

#### Auxiliary Loss
During the specialized routing phase, the router produces a vector of logits for each patch: $l \in \mathbb{R}^{N_{exp}}$, scaled by temperature: $\tilde{l} = \frac{l}{\tau}$, and transformed into probabilities via softmax: $p = \text{softmax}(\tilde{l})$

The project uses three auxiliary losses on the router: `Load Balancing Loss`, `Z-Loss`, and `Diversity Loss`.

---

**Load Balancing Loss** The `Load Balancing Loss` serves to prevent the router from consistently assigning too many patches to the same experts. For each expert $i$, we define:
$$f_i = \frac{1}{N} \sum_{j=1}^{N} m_{j,i}$$
where $m_{j,i}$ indicates whether patch $j$ was actually assigned to expert $i$ after the capacity constraint.

We then define:
$$P_i = \frac{1}{N} \sum_{j=1}^{N} p_{j,i}$$
where $p_{j,i}$ is the probability assigned by the router to expert $i$ for patch $j$.

The loss is:
$$\mathcal{L}_{bal} = N_{exp} \sum_{i=1}^{N_{exp}} f_i P_i$$
This loss combines the actual load of the experts, i.e., $f_i$, with the mean importance assigned by the router, i.e., $P_i$.

---

#### Z-Loss

The `Z-Loss` penalizes excessively large routing logits. It serves to stabilize the router by preventing the logits from growing too much and making the softmax overly sharp.

The formula used is:
$$\mathcal{L}_{z} =\frac{1}{N} \sum_{j=1}^{N} \left( \log \sum_{i=1}^{N_{exp}} e^{\tilde{l}_{j,i}} \right)^2$$
where $\tilde{l}_{j,i}$ is the temperature-scaled logit for patch $j$ and expert $i$.

---

#### Diversity Loss

The `Diversity Loss` serves to reduce redundancy among experts, pushing the router to produce less correlated activation patterns.
Given the probability matrix: $P \in \mathbb{R}^{N \times N_{exp}}$ where each row represents a patch and each column an expert, the code normalizes the columns of $P$: $\hat{P} = \text{normalize}(P, \text{dim}=0)$. Then it calculates the correlation matrix between experts: $C = \hat{P}^{T}\hat{P}$

The loss is:

$$\mathcal{L}_{div} = \frac{1}{N_{exp}^{2}} \sum_{i=1}^{N_{exp}} \sum_{j=1}^{N_{exp}} (C_{i,j} - I_{i,j})^2$$

where $I$ is the identity matrix. The goal is to make the correlation between experts closer to the identity: high on the diagonal, low off the diagonal. 

---

#### Total router loss

The total auxiliary router loss is: $\mathcal{L}_{router}=\alpha \mathcal{L}_{bal} + \beta \mathcal{L}_{z} + \lambda_{div} \mathcal{L}_{div}$

In the project, the weights are: $\beta = 10^{-4}$, $\lambda_{div} = 0.01$ while $\alpha$ is not fixed, but is scheduled during training. The total training loss becomes: $\mathcal{L}_{total}=\mathcal{L}_{CE}+\mathcal{L}_{router}$ therefore:

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \alpha \mathcal{L}_{bal} + 10^{-4}\mathcal{L}_{z} + 0.01\mathcal{L}_{div}$$

---

## Experiments
The conducted experiments aim to: 1) Measure the advantage of a ResNet-MoE compared to a Dense CNN. 2) Highlight the training dynamics (restricted to my setup) of the router in the context of hierarchical CNNs. For the first point, the final training accuracy and inference accuracy of the best MoE setup versus the dense setup were measured. For the second point, custom metrics were calculated to explain the router's behavior.

The MoE setups are distinguished by the number of available experts in the pool: 16 Experts, 8 Experts, and 4 Experts.

| Experts | Total Params | Active Params    | 
|---------|--------------|------------------|
| 4       | 33.5M        | 6M               |
| 8       | 58.8M        | 6M               |
| 16      | 109.6M       | 6M               | 

### Router Metrics 
To analyze the router's behavior, it is not sufficient to observe only the model's final accuracy. In fact, in MoE models, performance also depends on how traffic is distributed among the experts. For this reason, some custom routing metrics are tracked during training.

#### Spec Entropy

The `spec entropy` measures the average entropy of the probability distribution produced by the router before the Top-1 choice. For each patch, the router produces a distribution over the experts:
$$p = \text{softmax}\left(\frac{l}{\tau}\right)$$

The normalized entropy of the distribution is:
$$H_{spec}(p) = \frac{-\sum_{i=1}^{N_{exp}} p_i \log(p_i)}{\log(N_{exp})}$$

The final metric is the average over all patches:
$$\text{spec\_entropy} = \frac{1}{N}\sum_{j=1}^{N}H_{spec}(p_j)$$

High values indicate that the router produces more uniform distributions, meaning it is less confident in its choice of expert. Conversely, low values indicate sharper distributions, representing a more distinct selection.

---

#### Entropy Norm

The `entropy norm` measures how balanced the actual utilization of the experts is after dispatch. Let $u_i$ be the number of patches assigned to expert $i$. We define the normalized usage distribution:
$$q_i = \frac{u_i}{\sum_{k=1}^{N_{exp}} u_k}$$

The normalized entropy is:
$$H_{usage} = \frac{-\sum_{i=1}^{N_{exp}} q_i \log(q_i)}{\log(N_{exp})}$$

This metric assumes values between 0 and 1. A value close to 1 indicates that traffic is distributed across many experts. A value close to 0 indicates instead that traffic is concentrated on a few experts.

---

#### Imbalance Mean

The `imbalance mean` measures the imbalance between the most used and least used expert. Starting from the usage distribution $q_i$, it is calculated as:
$$\text{imbalance} = \frac{\max_i(q_i)}{\min_i(q_i) + \epsilon}$$
where $\epsilon$ is a small numerical stabilization term.

Low values indicate more balanced routing. High values indicate that some experts receive many more patches than others.

---

#### Drop Rate

The `drop rate` measures the fraction of patches that are not processed by the experts because the selected expert has already exceeded its capacity. Let $d_j = 1$ if patch $j$ is dropped and $d_j = 0$ otherwise. The drop rate is:
$$\text{drop\_rate} = \frac{1}{N}\sum_{j=1}^{N}d_j$$

A high drop rate would indicate that many tokens fail to enter the sparse path, meaning the experts' capacity would be a bottleneck.

---

#### Capacity Ratio

The `capacity ratio` measures how much of the available expert capacity is being utilized. Denoting the number of actually processed patches as $N_{processed}$ and the total available capacity as $N_{exp} \cdot C_{cap}$, the metric is:
$$\text{capacity\_ratio} = \frac{N_{processed}}{N_{exp} \cdot C_{cap}}$$

This metric helps determine whether the experts are working close to their maximum capacity or if a portion of the capacity remains unused.

---
### Results

**Accuracy & Latency**

<img src="../img/Best_MoEvsDense.png" width="500" align="left" style="margin-right: 20px; margin-bottom: 10px;">

| Model  | Top-1 Acc ↑ | Val CE ↓ | spec_entropy ↓ |
| ------ | ----------- | -------- | -------------- |
| Dense  | **63.85%** | 1.67     | --             |
| MoE-4  | 63.60%      | **1.66** | 0.50           |
| MoE-8  | 62.06%      | 1.76     | 0.64           |
| MoE-16 | 61.00%      | 1.78     | 0.68           |

<br clear="all">

The MoE model with 4 experts is the one that achieves the best accuracy among the sparse configurations. It is also the model with the lowest `spec_entropy`, and therefore the one where the router makes the sharpest decisions. However, it cannot be concluded that better local specialization of the router directly implies better accuracy: all MoE setups still show signs of overfitting. Furthermore, subsequent tests suggest that a significant portion of the performance also depends on the dense post-processing after the `rearrange`.

<img src="../img/MoE vs MoE.png" width="750" style="margin-right: 20px; margin-bottom: 10px;">


| Model      | Top-1 on inference subset (%) ↑ | Avg Latency (ms) ↓ | Std Latency (ms) ↓ |
| ---------- | ------------------------------- | ------------------ | ------------------ |
| Dense (18) | **67.33**                       | **2.07**           |**0.32**            |
| MoE-4      | 65.35                           | 27.61              | 4.47               |
| MoE-8      | 64.36                           | 35.76              | 4.48               |
| MoE-16     | 64.36                           | 47.57              | 4.33               |

<br clear="all">

**Router Metrics**

<img src="../img/MoE_metrics.png" width="600" align="left" style="margin-right: 20px; margin-bottom: 10px;">

| Model  | entr_norm | imbalance | drop_rate | cap_ratio |
| ------ | --------- | --------- | --------- | --------- |
| MoE-4  | 0.88      | 9.76      | 0.03      | 0.48      |
| MoE-8  | 0.92      | 24.19     | 0.02      | 0.48      |
| MoE-16 | 0.93      | 90.18     | 0.03      | 0.48      |

<br clear="all">

The router metrics show that as the number of experts increases, the router does not completely collapse: `entropy_norm` remains high, indicating that a large portion of the pool continues to be used. However, the `imbalance` grows significantly, going from 9.76 in the MoE-4 model to 90.18 in the MoE-16 model. This suggests that the main issue is not a total routing collapse, but an increasingly uneven load distribution among the experts.

**Overfitting**

During training, all MoE setups show signs of overfitting: the training loss continues to decrease, while the validation loss tends to worsen in the final stages. The cause is not entirely clear. Overfitting could depend on overly aggressive training, the router's difficulty in maintaining a stable load across experts, or an architectural cause related to the interaction between sparse and dense blocks.

Two main architectural hypotheses were tested.

<img src="../img/Refine_dense_blocks_MoE16.png" width="1000" style="margin-right: 20px; margin-bottom: 10px;">

1. **Artificial Receptive Field**: In the deeper layers, patches reach dimensions of 2 × 2. In this case, experts with 3 × 3 kernels could introduce an artificial receptive field relative to the actual patch size. The test was conducted by replacing the 3 × 3 kernels of the last MoE blocks with 1 × 1 kernels. This modification did not reduce overfitting.

2. **Dense Post-Processing**: The second hypothesis concerns the role of dense blocks after the sparse computation. An overly expressive dense post-processing might absorb part of the experts' capacity and become the main driver of backpropagation. The test was conducted by replacing the 3 × 3 kernel of the post-block after the `rearrange` with a 1 × 1 kernel. This modification also failed to eliminate overfitting, but it showed that the dense block is important for performance, given that its reduction worsens the Top-1 accuracy.

## Conclusion

A critical observation concerns the trend of the validation loss, which tends to diverge in the final stages of training despite a decreasing training loss. This overfitting, which is more pronounced as the number of experts increases, led us to formulate two architectural hypotheses.

The hierarchical MoE architecture on TinyImageNet demonstrates that higher sparsity capacity does not automatically translate into better generalization. The best regime is the one with 4 experts; larger configurations suffer from a load imbalance that the auxiliary loss function fails to compensate for. The observed overfitting does not seem to depend exclusively on local kernel dimensions or exclusively on the presence of dense post-processing, but appears primarily linked to the difficulty of stably optimizing traffic allocation among the experts.

All results can be reproduced by downloading the entire project repository and running the experimental script `main_experiments.py`, which calls the log analysis and model comparison functions. In the uploaded materials, this file is present as a dedicated script for the final experiments.