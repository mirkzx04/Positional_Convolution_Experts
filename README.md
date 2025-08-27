# PCE Network Implementation

## Idea

Traditional CNNs, even modern ones, apply the same filter across the entire image, which can lead to the loss of deep semantic details. Consider an image containing multiple people wearing different colored clothes, a street, trees, and an urban landscape (buildings, skyscrapers, etc.). Such an image is semantically rich. Trees have a well-defined semantics compared to other elements in the image. Applying the same filter uniformly may cause loss of details in the semantics of individual objects, such as the color of leaves or the precise texture of the pavement.

PCE (Patch-based Convolutional Experts) aims to solve this problem by applying convolutional experts to image patches that become progressively smaller as the network depth increases. This means that in deeper layers, experts specialize in very small patches, enabling them to learn semantic relationships within tiny portions of the image. These patches contain rich information about specific objects or parts of objects within the entire image. To decide which expert to assign a patch to, we use a router that takes as input the patch enriched with two spatial pieces of information: the absolute position of the patch within the image and the relative position of pixels within the patch.

## Architecture

The architecture consists of three main components:

### Router
- **Gate (Small MLP)**: Uses a two-layer MLP:
  
  $\text{logits} = W_2 \sigma(W_1 x + b_1) + b_2$
  where $\sigma = \text{GELU}$.
- **Expert Probabilities**:

  $p = \text{softmax}(\text{logits}) \in \mathbb{R}^E$
- **Top-1 Routing**:

  $e = \arg\max_e p_e, \quad g = \max_e p_e$
  That is, expert index $e$ and gate value $g \in [0,1]$.
- **Expert Capacity**:

  

  where $N = B \cdot P$ is the total number of patches in the batch and $E$ is the number of experts.

We use Top-1 routing with capacity limiting to restrict the number of patches served by each expert. Tokens beyond the capacity are dropped.
- **Auxiliary Losses**:
  - **Z-Loss**: To stabilize the logits:

    $\mathcal{L}\_{\text{z}} = \frac{1}{N} \sum\_{i=1}^{N} \left( \log \left( \sum\_{j=1}^{E} e^{l\_{i,j}} \right) \right)$
  - **Load Balancing Loss**:

    $\mathcal{L}\_{\text{balance}} = E \cdot \sum\_{e=1}^{E} \text{mean}(p\_e) \cdot \text{mean}(a\_e)$
    where $a_e$ is the allocation (number of patches assigned to expert $e$).

### Patch Extractor
- **Input**: $[B, C, H, W]$
- **Patch Division**: Extracts $P$ patches per image.
  If the patch has stride $S$, for an image it produces $h_{\text{patch}} = H/S$, $w_{\text{patch}} = W/S$, so $P = h_{\text{patch}} \cdot w_{\text{patch}}$.
- **Fourier Features**: Concatenates Fourier features to the channels, resulting in:

  $[B \cdot P, (C + C_{\text{fourier}}), H_{\text{patch}}, W_{\text{patch}}]$

### Experts
Double ResNet block without skip connections:

$\text{Conv}(K=3) \rightarrow \text{BatchNorm} \rightarrow \text{GELU} \rightarrow \text{Dropout} \rightarrow \text{Conv}(K=3) \rightarrow \text{BatchNorm}$

Each expert processes the assigned patches independently. The outputs are then combined based on the routing decisions.

This architecture allows experts to specialize in specific spatial and semantic regions, enhancing the network's ability to capture fine-grained details.