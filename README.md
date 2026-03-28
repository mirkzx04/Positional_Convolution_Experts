## Abstract

Studiamo un’architettura gerarchica Mixture-of-Experts (MoE) per la classificazione di immagini su TinyImageNet tramite routing Top-1 a livello di patch. Analizziamo l’impatto del numero di esperti (4, 8, 16) sulla generalizzazione, utilizzando metriche specifiche per monitorare sia la decisione locale del router sia il bilanciamento globale del carico tra esperti. I risultati mostrano che l’aumento della capacità sparsa degrada le performance di validazione ed è accompagnato da uno sbilanciamento del traffico, pur senza causare collassi del router né la comparsa di esperti morti. Infine, conduciamo alcune ablation per studiare la capacità locale degli esperti e l’impatto del post-processing denso, così da isolare le cause dell’overfitting osservato.

## Introduzione

L’impiego di architetture Mixture-of-Experts (MoE) nella computer vision mira a scalare la capacità del modello mantenendo invariato il costo computazionale attivo. Tuttavia, l’efficacia del routing appreso in strutture gerarchiche convoluzionali presenta sfide specifiche, legate alla specializzazione su token spaziali e alla stabilità del bilanciamento del carico.

In questo lavoro proponiamo un’architettura Vision MoE gerarchica custom testata su TinyImageNet (200 classi, risoluzione (224 \times 224)). Il contributo principale è un’analisi sistematica del comportamento del router al variare del pool di esperti. In particolare, distinguiamo tra due aspetti diversi del routing: da un lato la **decisione locale** del router sul singolo token, misurata tramite `spec_entropy`, dall’altro il **bilanciamento globale** del traffico tra esperti, misurato tramite `entropy_norm_mean`. Mostriamo che configurazioni MoE più ampie tendono a rendere il router meno deciso e contemporaneamente peggiorano la distribuzione del carico, con un effetto negativo sull’accuratezza Top-1. Investighiamo inoltre se l’overfitting osservato sia una conseguenza di un’eccessiva espressività locale degli esperti o dell’interazione tra routing sparse e blocchi densi residui. La struttura della rete, il router Top-1, le feature posizionali Fourier e le metriche aggregate di routing sono coerenti con l’implementazione del progetto.     

## Metodo

### Descrizione dell’architettura

L’architettura proposta riprende una struttura di tipo ResNet e la adatta a un setting MoE gerarchico. Ogni esperto è un blocco convoluzionale residuale del tipo

$$
\text{Conv}_{3 \times 3} \rightarrow \text{GN} \rightarrow \text{SiLU} \rightarrow \text{Conv}_{3 \times 3} \rightarrow \text{GN},
$$ 

seguito da una connessione residua interna all’esperto. I layer MoE sono alternati a blocchi densi di downsampling. Nel codice, gli esperti sono implementati come `ConvExpert`, mentre i blocchi di transizione sono implementati come `DownsampleResBlock`.  

Ogni layer MoE contiene (E) esperti, un’operazione di ricomposizione spaziale dei token processati (`rearrange`), una normalizzazione con GroupNorm, una SiLU e un blocco denso condiviso `post_block`. Indicando con (T_{in}) i token in ingresso e con (E_{out}) l’output aggregato degli esperti, la dinamica del layer può essere riassunta come:

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
res = post_block(moe\_out),
$$

$$
moe\_out = moe\_out + res,
$$

$$
X = X + moe_out.
$$

Qui $\text{alpha}$ è un parametro apprendibile del layer, mentre $X$ rappresenta la feature map passata al layer successivo. Il `post_block` condiviso è un blocco denso del tipo

$post_block = \text{Conv}_{3 \times 3} \rightarrow \text{GN} \rightarrow \text{SiLU}$ 

Questa struttura coincide con l’implementazione di `PCELayer` e con il flusso di forward di `PCENetwork`.  

Ad ogni blocco MoE, la feature map in ingresso viene scomposta in patch di dimensione (\text{patch_size} \times \text{patch_size}). Procedendo in profondità nella rete, il valore di `patch_size` viene ridotto nei punti di downsampling fino a un minimo di (2 \times 2). Le patch vengono inoltre arricchite con feature posizionali Fourier prima del routing.  

Il router riceve in input, per ogni patch, la concatenazione tra average pooling e max pooling spaziale del token arricchito:

$R_{in} = [\text{AVG}(T_{in}); \text{MAX}(T_{in})].$

Successivamente applica LayerNorm e una proiezione lineare per ottenere i logit degli esperti:

$l = \text{Linear}(\text{LN}(R_{in})).$

Le probabilità finali sono ottenute tramite

$S_{exp} = \text{Softmax}(l / \tau)$

e il routing seleziona l’esperto Top-1 soggetto a vincoli di capacità. Questa formulazione è coerente con `RouterGate` e `Router`.  

### Training

Il training è stato eseguito per 150 epoche con `batch_size = 128` su TinyImageNet. L’ottimizzatore utilizzato è `AdamW`, con gruppi di parametri separati per backbone e router, learning rate distinti e scheduler di tipo warmup + cosine annealing. Nel codice il router è anche escluso dal weight decay. Il training set usa una pipeline di data augmentation forte che include `RandomResizedCrop`, `RandomHorizontalFlip`, `RandomGrayscale`, `ColorJitter`, `RandAugment`, `RandomErasing`, oltre a `MixUp` e `CutMix` applicati nel training step.  

Oltre alla cross-entropy del task, vengono impiegate tre loss ausiliarie sul router:

* **Load Balancing Loss**:

  $L_{bal} = E \sum_{i=1}^{E} f_i \cdot P_i$

  dove (f_i) è la frazione di token assegnati all’esperto (i) e (P_i) è la probabilità media assegnata a quell’esperto.

* **Z-Loss**:

  $L_z = \frac{1}{N}\sum_j \left(\log \sum_i e^{l_i(x_j)}\right)^2$

  che penalizza logit di magnitudo troppo elevata.
* **Diversity Loss**, che disincentiva pattern di attivazione ridondanti tra esperti minimizzando la correlazione tra le probabilità di routing. 

## Esperimenti e risultati

### Variazione del pool di esperti

Analizziamo tre configurazioni MoE (4, 8, 16 esperti) rispetto a una baseline densa. La tabella mostra che la configurazione MoE-4 è la più performante, raggiungendo il $63.6%$ di Top-1 accuracy, valore molto vicino al modello denso $63.85%$ Tuttavia, aumentando il numero di esperti, le performance degradano progressivamente.

La metrica `spec_entropy` misura l’entropia normalizzata della distribuzione di probabilità del router **prima del dispatch**, mediata sui token: valori più alti indicano un router meno deciso nella scelta locale dell’esperto. Al contrario, `entropy_norm_mean` misura l’entropia normalizzata della distribuzione di utilizzo degli esperti **dopo il dispatch**, quindi descrive quanto il traffico complessivo sia globalmente distribuito in modo uniforme nel pool. In altre parole, `spec_entropy` cattura l’incertezza locale del router, mentre `entropy_norm_mean` cattura il bilanciamento globale del carico. Questa distinzione è esattamente quella implementata nel `MoEAggregator`. 

| Modello | Top-1 Acc ↑ | Val CE ↓ | spec_entropy ↓ |
| ------- | ----------- | -------- | -------------- |
| Dense   | **63.85%**  | 1.67     | --             |
| MoE-4   | 63.60%      | **1.66** | 0.50           |
| MoE-8   | 62.06%      | 1.76     | 0.64           |
| MoE-16  | 61.00%      | 1.78     | 0.68           |

Per completezza, riportiamo anche il costo inferenziale medio delle diverse configurazioni. In questa tabella, la colonna Top-1 è calcolata su un **dataset di inferenza dedicato**, ottenuto come sottoinsieme del validation set, mentre latenza media e deviazione standard sono misurate nel medesimo setting. Nel regime considerato, l’introduzione dello sparse routing non produce un vantaggio di efficienza pratica.

| Modello    | Top-1 su subset inferenza (%) ↑ | Avg Latency (ms) ↓ | Std Latency (ms) ↓ |
| ---------- | ------------------------------- | ------------------ | ------------------ |
| Dense (18) | **67.33**                       | **2.07**           | **0.32**           |
| MoE-4      | 65.35                           | 27.61              | 4.47               |
| MoE-8      | 64.36                           | 35.76              | 4.48               |
| MoE-16     | 64.36                           | 47.57              | 4.33               |

### Analisi delle metriche di routing

L’analisi delle metriche di dispatching mostra che il router non collassa, ma diventa progressivamente più difficile da bilanciare all’aumentare del numero di esperti.

Primo, `entropy_norm_mean` resta elevata, segnalando che il pool di esperti rimane nel complesso attivo. Tuttavia, questa metrica da sola non basta a garantire un buon bilanciamento. Infatti, `imbalance_mean` cresce drasticamente da $9.76$ a $90.18$ nel caso con 16 esperti, indicando che alcuni esperti ricevono molto più traffico di altri. Quindi il problema non è un collasso totale del routing, ma un’assegnazione fortemente sbilanciata.

Secondo, il `drop_rate` resta quasi costante attorno a $0.02$ – $0.03$, e il `cap_ratio` rimane stabile, suggerendo che il degrado non dipenda principalmente da colli di bottiglia di capacità o da un eccessivo numero di token scartati.

Terzo, non emergono **esperti morti**: nei run analizzati il comportamento è coerente con `dead_mean \approx 0`, mentre `active_mean` resta elevata. Questo è importante perché indica che il problema non è l’inattività completa di parte del pool, ma piuttosto una forte sotto-utilizzazione relativa di alcuni esperti rispetto ad altri. Le metriche `dead_mean` e `active_mean` sono esplicitamente previste nel sistema di aggregazione del progetto. 

| Modello | entr_norm | imbalance | drop_rate | cap_ratio |
| ------- | --------- | --------- | --------- | --------- |
| MoE-4   | 0.88      | 9.76      | 0.03      | 0.48      |
| MoE-8   | 0.92      | 24.19     | 0.02      | 0.48      |
| MoE-16  | 0.93      | 90.18     | 0.03      | 0.48      |

## Discussione e conclusioni

Un’osservazione critica riguarda l’andamento della validation loss, che tende a divergere nelle fasi finali del training nonostante una training loss decrescente. Questo overfitting, più marcato all’aumentare del numero di esperti, ci ha portato a formulare due ipotesi architetturali.

**1. Capacità locale degli esperti.** Nei layer profondi, dove le patch raggiungono dimensioni $2 \times 2$, esperti con kernel $3 \times 3$ potrebbero introdurre un campo recettivo artificiale. Tuttavia, un run sperimentale in cui i kernel $3 \times 3$ sono stati sostituiti con $1 \times 1$ negli ultimi due layer MoE non ha ridotto l’overfitting, indebolendo questa ipotesi.

**2. Post-processing denso.** La sequenza residuale che include il `post_block` denso potrebbe assorbire troppa capacità, correggendo o sovrascrivendo parte dell’output prodotto dagli esperti sparse. Per verificarlo abbiamo modificato tale componente, osservando però un peggioramento sia della `validation_class_loss` sia della Top-1 accuracy. Questo suggerisce che il post-processing denso resti una componente utile per la generalizzazione, pur non eliminando il fenomeno di overfitting. La struttura del `post_block` e dei layer coinvolti è coerente con il codice del modello.  

In conclusione, l’architettura MoE gerarchica su TinyImageNet mostra che una maggiore capacità sparsa non si traduce automaticamente in una migliore generalizzazione. Il regime migliore è quello con 4 esperti; configurazioni più ampie soffrono di uno sbilanciamento del carico che la loss ausiliaria non riesce a compensare. L’overfitting osservato non sembra dipendere unicamente dalla dimensione dei kernel locali né dalla sola presenza del post-processing denso, ma appare soprattutto legato alla difficoltà di ottimizzare in modo stabile l’assegnazione del traffico tra esperti.

Tutti i risultati sono rieseguibili scaricando l’intera repository del progetto e lanciando lo script sperimentale `main_experiments.py`, che richiama le funzioni di analisi dei log e dei confronti tra modelli. Nel materiale che hai caricato questo file è presente come script dedicato agli esperimenti finali. 

