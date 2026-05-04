## Abstract

This project studies the training dynamics of a Vision Mixture-of-Experts (MoE) architecture in a setting different from classic Transformer-based models.

Specifically, the project analyzes an MoE embedded within a hierarchical CNN. Unlike Vision Transformers, where tokens interact globally through attention, a hierarchical CNN operates with a local and progressively downsampled structure. This makes MoE routing over patches an interesting case: the router must specialize experts using local and positional information, without directly benefiting from global attention.

The experiments suggest that, under the current setup, a CNN-based MoE may suffer from load imbalance in the assignments made by the router to the experts. This imbalance appears to correlate with degraded generalization, but the cause of the overfitting is not completely isolated. In fact, additional tests show that dense residual blocks also have a strong impact on performance, leaving the relationship between sparse routing, dense post-processing, imbalance, and overfitting as an open question.

---

## Introduction

In this work, an MoE architecture based on a hierarchical CNN is proposed. The convolutional component resides in the model backbone, while sparse routing is applied at the patch level on intermediate feature maps.

The model was trained on TinyImageNet, which consists of 200 classes and images resized to 224 × 224. The main contribution of this project is the analysis of the routing dynamics. In particular, two aspects are distinguished:

1. **Local router decision**, measured through `spec_entropy`, which indicates how sharp or uniform the probability distribution over experts is.
2. **Global traffic balancing**, measured through `entropy_norm_mean` and `imbalance_mean`, which describe how patches are distributed across experts.

---

## Method

### Architecture

![PCE Architecture](../img/PCE_architecture.png)

### MoE Blocks

The MoE blocks operate at the patch level on the feature map. Each input feature map is divided by the `Patch Extractor` into patches of size $P_s \times P_s$. For each patch, positional features are computed using `Fourier Positional Features`, concatenated to the patch content, and used exclusively by the router.

The router therefore receives patches enriched with positional information and decides, via Top-1 routing, which expert should process each patch. The experts, on the other hand, receive only the original patch content, without positional embeddings. In this way, spatial specialization is delegated to the router: it is the router that, by observing both the content and the position of the patch in the original feature map, must select the most suitable expert.

After routing, each convolutional expert processes only the patches assigned to it. The experts' outputs are then weighted by the combination coefficient produced by the router and accumulated back into the position corresponding to the original patch.

Denoting the aggregated output of the experts by $E_{\text{out}}$ and the original input patches by $X_p$, the first residual is defined as:

$$
O = X_p + \gamma E_{\text{out}}
$$

where $\gamma$ is a learnable parameter initialized to a small value. This residual serves two main functions: preserving the original signal during the early stages of training and making the experts' contribution gradual, preventing sparse routing from destabilizing the representation too early.

At this point, the processed patches are recomposed into their original spatial arrangement via `rearrange`:

$$
R_{\text{out}}(O) \in \mathbb{R}^{B \times C_{\text{out}} \times H_{\text{out}} \times W_{\text{out}}}
$$

The reconstructed feature map is then normalized and activated:

$$
\operatorname{MoE}_{\text{out}} = \operatorname{SiLU}\left( \operatorname{GN}\left( R_{\text{out}}(O) \right) \right)
$$

This step stabilizes the feature distribution after the sparse processing and before the subsequent dense phase.

The third block is a shared convolutional block:

$$
\operatorname{res} = \operatorname{Conv}_{3 \times 3} \rightarrow \operatorname{GN} \rightarrow \operatorname{SiLU}
$$

Its role is to locally mix the features produced by the experts. This is important because the patches are processed separately and then reinserted into their original positions: without a local dense operation, discontinuities could emerge between adjacent patches or overly sharp boundaries between regions processed by different experts.

Finally, the layer applies two residual connections:

$$
\operatorname{MoE}_{\text{out}} = \operatorname{MoE}_{\text{out}} + \operatorname{res}
$$

$$
X_{\text{out}} = R_{\text{out}}(O) + \operatorname{MoE}_{\text{out}}
$$

The first residual integrates the contribution of the dense convolutional block, while the second maintains a direct connection with the reconstructed feature map after expert processing. In this way, the block combines three components: sparse routing across experts, shared normalization/activation, and dense convolutional post-processing.

### Router Gate

The `Router Gate` is the module that assigns each patch to one of the available experts. For each patch $X_p$, the gate constructs a compact representation using global spatial statistics:

$$
R_{\text{in}} = \left[ \operatorname{mean}(X_p); \operatorname{amax}(X_p) \right]
$$

where $\operatorname{mean}(X_p)$ and $\operatorname{amax}(X_p)$ are computed along the spatial dimensions of the patch. In this way, each patch is represented by a vector that summarizes both its mean activation and its maximum activation.

This representation is normalized and passed through a linear layer:

$$
\ell = W \cdot \operatorname{LN}(R_{\text{in}}) + b
$$

where $\ell \in \mathbb{R}^{N_{\text{exp}}}$. The vector $\ell$ contains one logit for each expert. The `Router Gate` can therefore be seen as a function that, given the representation of a patch, assigns a score to each expert in the pool.

The logits are then scaled by a temperature $\tau$ and transformed into probabilities through softmax:

$$
p_{\text{exp}} = \operatorname{softmax}\left( \frac{\ell}{\tau} \right)
$$

The temperature controls how uniform or selective the distribution over experts is: higher values of $\tau$ produce softer distributions, while lower values make the logits sharper and push the router toward more distinct choices.

Once the distribution over the experts is obtained, Top-1 routing is applied: for each patch, the expert with the highest probability is selected:

$$
e^* = \operatorname*{arg\,max}_i \, p_i
$$

However, each expert has a maximum capacity, denoted by `capacity`, which limits the number of patches it can process in a single forward pass. The capacity is computed based on the total number of patches, the number of experts, and the `capacity_factor`:

$$
C_{\text{cap}} = \left\lceil \frac{\text{capacity\_factor} \cdot N}{N_{\text{exp}}} \right\rceil
$$

where $N$ is the total number of patches to be routed. If an expert receives more patches than its capacity, only the patches with the highest routing probability are retained, while the others are dropped from the sparse path.

The dropped patches are not processed by the experts; however, thanks to the residual connection in the MoE block, the original patch signal can still be preserved in the layer. Capacity therefore serves to control the experts' computational load and to prevent a single expert from absorbing all the traffic.

---

## Training

The model was trained on TinyImageNet with images resized to 224 × 224. The following setup was used for all training sessions:

| Learning Rate Backbone | Learning Rate Router | Batch Size | Stem Kernel | Stem Out Channels | Optimizer | Weight Decay |
|------------------------|----------------------|------------|-------------|-------------------|-----------|--------------|
| 0.001                  | 0.001                | 128        | 7x7         | 64                | AdamW     | 1e-3         |

In this setup, the backbone and router share the optimizer but not the learning rate. The choice of a higher learning rate for the router stems from the fact that it must adapt quickly to the features extracted by the backbone. Since these features are already enriched by expert computation and residual connections, a learning rate that is too low could prevent the router from adapting fast enough. Furthermore, the router is exempt from weight decay.

Both the backbone and router learning rates go through a warmup phase:

1. **Backbone Warmup**: occurs during the first training epochs.
2. **Router Warmup**: in the early training epochs, the router assigns patches uniformly to all experts until each expert reaches capacity. During this phase, the router is completely frozen and receives no backpropagation; after this phase, it goes through its own warmup.

| Tau Init | Tau Final | Aux Loss Weight Init | Aux Loss Final |
|----------|-----------|----------------------|----------------|
| 2.0      | 0.85      | 0.05                 | 5e-4           |

The router is also governed by additional hyperparameters: `tau`, which regulates the softmax temperature over the logits, and `aux_loss_weight`, which regulates the weight of the load-balancing loss.

The scheduling of the auxiliary coefficient $\alpha$ is defined as:

$$
\alpha =
\begin{cases}
0, & e < e_{\text{router}} \\
\alpha_{\text{peak}} \cdot \frac{e - e_{\text{router}} + 1}{e_{\text{warmup}}}, & e_{\text{router}} \le e < e_{\text{router}} + e_{\text{warmup}} \\
\alpha_{\text{peak}}, & e_{\text{router}} + e_{\text{warmup}} \le e < e_{\text{decay}} \\
\alpha_{\text{final}} + (\alpha_{\text{peak}} - \alpha_{\text{final}}) \cdot \frac{1 + \cos(\pi t)}{2}, & e \ge e_{\text{decay}}
\end{cases}
$$

In addition, routing noise is injected into the logits for exploration. All of these hyperparameters are scheduled during training, and the routing noise reaches 0.0 at the end of training.

### Auxiliary Loss

During the specialized routing phase, the router produces a vector of logits for each patch, $\ell \in \mathbb{R}^{N_{\text{exp}}}$. These logits are temperature-scaled as:

$$
\tilde{\ell} = \frac{\ell}{\tau}
$$

and transformed into probabilities via softmax:

$$
p = \operatorname{softmax}(\tilde{\ell})
$$

The project uses three auxiliary losses on the router: `Load Balancing Loss`, `Z-Loss`, and `Diversity Loss`.

---

### Load Balancing Loss

The `Load Balancing Loss` serves to prevent the router from consistently assigning too many patches to the same experts. For each expert $i$, define:

$$
f_i = \frac{1}{N} \sum_{j=1}^{N} m_{j,i}
$$

where $m_{j,i}$ indicates whether patch $j$ was actually assigned to expert $i$ after the capacity constraint.

Then define:

$$
P_i = \frac{1}{N} \sum_{j=1}^{N} p_{j,i}
$$

where $p_{j,i}$ is the routing probability assigned to expert $i$ for patch $j$.

The load-balancing loss is:

$$
\mathcal{L}_{\text{bal}} = N_{\text{exp}} \sum_{i=1}^{N_{\text{exp}}} f_i P_i
$$

This loss combines the actual load of the experts, represented by $f_i$, with the mean importance assigned by the router, represented by $P_i$.

---

### Z-Loss

The `Z-Loss` penalizes excessively large routing logits. Its purpose is to stabilize the router by preventing the logits from growing too much and making the softmax overly sharp.

The formula is:

$$
\mathcal{L}_z = \frac{1}{N} \sum_{j=1}^{N} \left( \log \sum_{i=1}^{N_{\text{exp}}} e^{\tilde{\ell}_{j,i}} \right)^2
$$

where $\tilde{\ell}_{j,i}$ is the temperature-scaled logit for patch $j$ and expert $i$.

---

### Diversity Loss

The `Diversity Loss` serves to reduce redundancy among experts, encouraging the router to produce less correlated activation patterns.

Given the probability matrix:

$$
P \in \mathbb{R}^{N \times N_{\text{exp}}}
$$

where each row represents a patch and each column an expert, the code normalizes the columns of $P$:

$$
\hat{P} = \operatorname{normalize}(P, \mathrm{dim}=0)
$$

It then computes the expert correlation matrix:

$$
C = \hat{P}^{\top} \hat{P}
$$

The diversity loss is defined as:

$$
\mathcal{L}_{\text{div}} = \frac{1}{N_{\text{exp}}^2} \sum_{i=1}^{N_{\text{exp}}} \sum_{j=1}^{N_{\text{exp}}} \left( C_{i,j} - I_{i,j} \right)^2
$$

where $I$ is the identity matrix. The goal is to make the correlation matrix as close as possible to the identity: high on the diagonal and low off the diagonal.

---

### Total Router Loss

The total auxiliary router loss is:

$$
\mathcal{L}_{\text{router}} = \alpha \mathcal{L}_{\text{bal}} + \beta \mathcal{L}_z + \lambda_{\text{div}} \mathcal{L}_{\text{div}}
$$

In the project, the weights are:

$$
\beta = 10^{-4}, \qquad \lambda_{\text{div}} = 0.01
$$

while $\alpha$ is scheduled during training rather than fixed.

The total training loss is therefore:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{router}}
$$

that is:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \alpha \mathcal{L}_{\text{bal}} + 10^{-4}\mathcal{L}_z + 0.01\mathcal{L}_{\text{div}}
$$

---

## Experiments

The experiments were designed to:

1. measure the advantage of a ResNet-MoE compared to a dense CNN;
2. highlight the router training dynamics, restricted to this specific setup, in the context of hierarchical CNNs.

For the first point, the final training accuracy and inference accuracy of the best MoE setup were compared against the dense setup. For the second point, custom routing metrics were computed to better explain router behavior.

The MoE setups are distinguished by the number of experts available in the pool: 4 experts, 8 experts, and 16 experts.

| Experts | Total Params | Active Params |
|---------|--------------|---------------|
| 4       | 33.5M        | 6M            |
| 8       | 58.8M        | 6M            |
| 16      | 109.6M       | 6M            |

---

## Router Metrics

To analyze the router's behavior, it is not sufficient to observe only the model's final accuracy. In MoE models, performance also depends on how traffic is distributed among experts. For this reason, several custom routing metrics are tracked during training.

### Spec Entropy

The `spec_entropy` measures the average entropy of the probability distribution produced by the router before the Top-1 selection. For each patch, the router produces a distribution over experts:

$$
p = \operatorname{softmax}\left( \frac{\ell}{\tau} \right)
$$

The normalized entropy of the distribution is:

$$
H_{\text{spec}}(p) = \frac{-\sum_{i=1}^{N_{\text{exp}}} p_i \log(p_i)}{\log(N_{\text{exp}})}
$$

The final metric is the average over all patches:

$$
\text{spec\_entropy} = \frac{1}{N} \sum_{j=1}^{N} H_{\text{spec}}(p_j)
$$

High values indicate that the router produces more uniform distributions, meaning it is less confident in expert selection. Low values indicate sharper distributions and therefore more distinct choices.

---

### Entropy Norm

The `entropy_norm` measures how balanced expert utilization is after dispatch. Let $u_i$ be the number of patches assigned to expert $i$. Define the normalized usage distribution as:

$$
q_i = \frac{u_i}{\sum_{k=1}^{N_{\text{exp}}} u_k}
$$

The normalized entropy is:

$$
H_{\text{usage}} = \frac{-\sum_{i=1}^{N_{\text{exp}}} q_i \log(q_i)}{\log(N_{\text{exp}})}
$$

This metric takes values between 0 and 1. A value close to 1 indicates that traffic is distributed across many experts, while a value close to 0 indicates that traffic is concentrated on only a few experts.

---

### Imbalance Mean

The `imbalance_mean` measures the imbalance between the most used and least used experts. Starting from the usage distribution $q_i$, it is computed as:

$$
\text{imbalance} = \frac{\max_i q_i}{\min_i q_i + \varepsilon}
$$

where $\varepsilon$ is a small numerical stabilization constant.

Low values indicate more balanced routing. High values indicate that some experts receive many more patches than others.

---

### Drop Rate

The `drop_rate` measures the fraction of patches that are not processed by the experts because the selected expert has already exceeded its capacity. Let:

$$
d_j =
\begin{cases}
1, & \text{if patch } j \text{ is dropped} \\
0, & \text{otherwise}
\end{cases}
$$

Then the drop rate is:

$$
\text{drop\_rate} = \frac{1}{N} \sum_{j=1}^{N} d_j
$$

A high drop rate indicates that many tokens fail to enter the sparse path, meaning that expert capacity is acting as a bottleneck.

---

### Capacity Ratio

The `capacity_ratio` measures how much of the available expert capacity is actually used. Denoting the number of processed patches by $N_{\text{processed}}$ and the total available capacity by $N_{\text{exp}} \cdot C_{\text{cap}}$, the metric is:

$$
\text{capacity\_ratio} = \frac{N_{\text{processed}}}{N_{\text{exp}} \cdot C_{\text{cap}}}
$$

This metric helps determine whether experts are working close to maximum capacity or whether part of the available capacity remains unused.

---

## Results

### Accuracy and Latency

![Best MoE vs Dense](../img/Best_MoEvsDense.png)

| Model  | Top-1 Acc ↑ | Val CE ↓ | spec_entropy ↓ |
|--------|-------------|----------|----------------|
| Dense  | **63.85%**  | 1.67     | --             |
| MoE-4  | 63.60%      | **1.66** | 0.50           |
| MoE-8  | 62.06%      | 1.76     | 0.64           |
| MoE-16 | 61.00%      | 1.78     | 0.68           |

The MoE model with 4 experts achieves the best accuracy among the sparse configurations. It is also the model with the lowest `spec_entropy`, and therefore the one in which the router makes the sharpest decisions. However, this does not imply that better local specialization directly leads to better accuracy: all MoE setups still show signs of overfitting. Moreover, subsequent tests suggest that a significant portion of the final performance also depends on the dense post-processing after the `rearrange`.

![MoE vs MoE](../img/MoE%20vs%20MoE.png)

| Model      | Top-1 on inference subset (%) ↑ | Avg Latency (ms) ↓ | Std Latency (ms) ↓ |
|------------|---------------------------------|--------------------|--------------------|
| Dense (18) | **67.33**                       | **2.07**           | **0.32**           |
| MoE-4      | 65.35                           | 27.61              | 4.47               |
| MoE-8      | 64.36                           | 35.76              | 4.48               |
| MoE-16     | 64.36                           | 47.57              | 4.33               |

### Router Metrics

![MoE Metrics](../img/MoE_metrics.png)

| Model  | entr_norm | imbalance | drop_rate | cap_ratio |
|--------|-----------|-----------|-----------|-----------|
| MoE-4  | 0.88      | 9.76      | 0.03      | 0.48      |
| MoE-8  | 0.92      | 24.19     | 0.02      | 0.48      |
| MoE-16 | 0.93      | 90.18     | 0.03      | 0.48      |

The router metrics show that, as the number of experts increases, the router does not completely collapse: `entropy_norm` remains high, indicating that a large portion of the expert pool continues to be used. However, the `imbalance` grows significantly, from 9.76 in the MoE-4 model to 90.18 in the MoE-16 model. This suggests that the main issue is not total routing collapse, but rather an increasingly uneven load distribution across experts.

---

## Overfitting

During training, all MoE setups show signs of overfitting: the training loss continues to decrease, while the validation loss tends to worsen in the final stages. The cause is not entirely clear. Overfitting may depend on overly aggressive training, on the router's difficulty in maintaining stable load allocation across experts, or on an architectural interaction between sparse and dense blocks.

Two main architectural hypotheses were tested.

![Refine Dense Blocks MoE16](../img/Refine_dense_blocks_MoE16.png)

1. **Artificial Receptive Field**: in the deeper layers, patches reach dimensions of 2 × 2. In this case, experts with 3 × 3 kernels could introduce an artificial receptive field relative to the actual patch size. This hypothesis was tested by replacing the 3 × 3 kernels of the last MoE blocks with 1 × 1 kernels. This modification did not reduce overfitting.

2. **Dense Post-Processing**: the second hypothesis concerns the role of dense blocks after sparse computation. An overly expressive dense post-processing stage might absorb part of the experts' capacity and become the main driver of backpropagation. This was tested by replacing the 3 × 3 kernel of the post-block after the `rearrange` with a 1 × 1 kernel. This modification also failed to eliminate overfitting, but it showed that the dense block is important for performance, since reducing it worsened Top-1 accuracy.

---

## Conclusion

A critical observation concerns the trend of the validation loss, which tends to diverge in the final stages of training despite a decreasing training loss. This overfitting, which becomes more pronounced as the number of experts increases, motivated the formulation of two architectural hypotheses.

The hierarchical MoE architecture on TinyImageNet shows that increasing sparse capacity does not automatically translate into better generalization. The best regime is the one with 4 experts, while larger configurations suffer from load imbalance that the auxiliary loss function fails to compensate for. The observed overfitting does not seem to depend exclusively on local kernel dimensions, nor exclusively on the presence of dense post-processing, but appears primarily linked to the difficulty of stably optimizing traffic allocation across experts.

All results can be reproduced by downloading the full project repository and running the experimental script `main_experiments.py`, which calls the log analysis and model comparison functions. In the uploaded materials, this file is included as a dedicated script for the final experiments.