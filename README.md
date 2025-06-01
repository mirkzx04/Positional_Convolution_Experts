# Progetto: Positional Convolution Experts

## Abstract

Le CNN tradizionali processano le immagini attraverso un singolo canale convoluzionale, limitando potenzialmente la capacità della rete di catturare informazioni posizionali specifiche. Questo progetto propone un'architettura innovativa basata su **Positional Convolution Experts** che sfrutta sia il contenuto che la posizione delle patch per indirizzare ciascuna patch verso esperti specializzati, ottenendo feature map più ricche e rappresentative.

## Motivazione

Il problema principale delle CNN tradizionali risiede nella loro incapacità di sfruttare efficacemente le informazioni posizionali delle patch all'interno dell'immagine. Dividendo l'immagine in patch e utilizzando sia contenuto che posizione per il routing verso esperti specializzati, ogni esperto può focalizzarsi su caratteristiche specifiche di regioni specifiche dell'immagine.

## Architettura Proposta

### Concetto Fondamentale

L'architettura si basa su un sistema di routing intelligente che:
- Divide l'immagine in patch
- Utilizza informazioni di contenuto e posizione per il routing
- Indirizza ciascuna patch verso esperti specializzati
- Combina i risultati per ottenere feature map generali arricchite

### Pipeline approfondita

L'immagine di input nella rete $[B,C,H,W]$ dove:
- B -> Batch size
- C -> Channel (RGB)
- H -> Height
- W -> Width

La patch viene divisa in patch di dimensioni $hP \times wP$, applicando poi CoordConv su pixel-level aggiungiamo informazione spaziale su dove si trovano i pixel all'interno della patch e dove si trova la patch rispetto all'immagine, ottenendo 
$[B, nP, C +4, H, W]$ dove $C+4$ rappresenta l'aggiunta delle coordinate della patch e di ogni pixel all'interno dei patch.
Le patch vengono usate prima del training per inizializzare delle chiavi all'interno del router.

Le chiavi vengono inizializzate tramite una convoluzione $1 \times 1$ sulla patch ridimensionata come $$[B \times P, C+4, H, W]$$, il risultato di questa convoluzione viene passato a SSP per produrre dei patch embedding $ Em_p $ che poi saranno applicati a K-Means per ottenere i centroidi che verranno usati come chiavi $k \in \mathbb{R^{n_{exp} \times d}}$ dove $D = (C+4) \times (1^2 + 2^2 4^2)$.

Il router applicherà cosine similarity tra $Em_p$ e $k$ applicando poi softmax per ottenere le probabilità di scelta dei diversi esperti.
Gli esperti convoluzionali son definiti ocome $Conv_{kz \times kz} - BatchNorm - ReLU$ produrranno diverse feature map che verranno concatenate attraverso una somma pesata, dove i pesi saranno gli score delle probabilità date dalla softmax.

Ottenuta la feature map globale questa verrà ridivisa in patch nel modo descritto sopra e le patch verranno riapplicate al router.

## Metodologia di Training

### Fase 1: Stabilizzazione dei Parametri

Tre approcci alternativi per la fase iniziale:

#### Opzione A: Routing Deterministico con Rumore
- Routing basato sulla posizione della patch con aggiunta di rumore controllato
- Gli esperti si specializzano mantenendo un mix causato dal rumore
- **Vantaggi**: Specializzazione garantita con diversificazione

#### Opzione B: Cosine Similarity con Chiavi EMA
Utilizza cosine similarity con chiavi inizializzate tramite SSP per la standardizzazione della dimensionalità mantenendo le relazioni spaziali pixel-level e, K-Means per clusterizzare le patch attorno a dei centroidi che poi verranno usati come chiavi per calcolare la similarità con l'embedding della patch ottenuti tramite SSP.
La patch verrà passata in canale SSP in modo da produrre un embedding della patch $ Em_p $ che verrà usato per il calcolo della similarità.
Le chiavi rappresentano medie mobili assegnate a ciascuna patch, mantenute fuori dal grafo computazionale.

#### Opzione C: Distribuzione Uniforme
Utilizzo di una distribuzione fittizia:

$w_i = \frac{\text{numero patch}}{\text{numero esperti}}$

### Fase 2: Introduzione del Router

#### Router MLP
**Caratteristiche:**
- Più computazionalmente pesante
- Richiede backpropagation completa
- Pre-processing con CoordConv per arricchire le patch con informazioni posizionali

**Pipeline:**
1. Applicazione di CoordConv alla patch
2. Feature map arricchita → MLP
3. Output: distribuzione di probabilità sugli esperti
4. Somma ponderata delle feature map degli esperti

#### Key Attention Routing

**Processo:**

1. **Formazione del vettore chiave:**
   $k \in \mathbb{R}^d$

2. **Embedding della patch:**
   $p \in \mathbb{R}^{C \times H \times W}$

**Inizializzazione delle chiavi:**
Standardizzazione delle dimensioni con SSP e clusterizzazione con K-Means

**Calcolo della similarità:**

$s_i = \frac{Em_p \cdot k}{||Em_p \cdot k||}$

**Pesi di routing:**

$w_j = \frac{e^{s_j}}{\sum_j e^{s_j}}$ o softmax

**Output finale:**

$\text{out} = \sum_i w_i \cdot E_i(p)$

**Aggiornamento delle chiavi:**
- nn.Parameter di PyTorch (Probabilmente applicata dopo una fase di stabilizzazione con EMA) per ottimizzare le chiavi tramite backpropagation
- EMA con parametri backpropagabili $k_i^{t+1} = \alpha \cdot k_i^t + (1 - \alpha) \cdot v_i$

#### Gumbel Softmax
Mantiene l'approccio dell'Opzione B nella fase di pre-training con campionamento differenziabile.

### Fase 3: Introduzione dell'Attention

**Pipeline:**
1. Somma di concatenazione → Convoluzione 1×1
2. Feature map risultante → Modulo di Attention
3. Feature map arricchita per il routing successivo

**Valutazione:** Le performance vengono monitorate per determinare l'utilità del modulo.

## Valutazione e Metriche

### Object Detection
**Dataset:** CIFAR-10, Tiny-ImageNet, Pascal VOC

**Metriche:**
- **Accuracy:** Top-1 e Top-5 su CIFAR-10 e Tiny-ImageNet
- **mAP:** Mean Average Precision su Pascal VOC

### Segmentazione
**Dataset:** Pascal VOC, Camelyon

**Metriche:**
- **mIoU:** Mean Intersection over Union

## Risultati Attesi

### Performance
- Miglioramento dell'accuracy rispetto alle CNN tradizionali
- Performance competitive con le CNN moderne
- Specializzazione degli esperti basata sulla posizione delle patch

### Interpretabilità
Le rotte di routing offrono insights unici:
- **Analisi delle assegnazioni:** Quale patch viene assegnata a quale esperto
- **Specializzazione degli esperti:** Comprensione delle caratteristiche apprese
- **Ottimizzazione dell'architettura:** Rimozione o rinforzo di esperti basato sull'entropia del router

### Metriche di Analisi
- **Entropia del router:** Misura della distribuzione delle assegnazioni
- **Clustering delle rotte:** Analisi dei pattern di routing
- **Statistiche degli esperti:** Utilizzo medio e specializzazione

## Vantaggi dell'Approccio

1. **Specializzazione Posizionale:** Ogni esperto si focalizza su regioni specifiche
2. **Feature Map Arricchite:** Combinazione di informazioni locali e globali
3. **Interpretabilità:** Possibilità di analizzare le decisioni di routing
4. **Flessibilità:** Multipli approcci di training e routing
5. **Ottimizzazione Adattiva:** Possibilità di modificare la rete basandosi sull'analisi delle rotte

## Implementazione

Il progetto sarà implementato utilizzando PyTorch, con particolare attenzione a:
- Modularità del codice per testare diverse opzioni di routing
- Logging dettagliato per l'analisi delle performance
- Visualizzazione delle rotte di routing per l'interpretabilità
- Benchmarking sistematico contro architetture baseline