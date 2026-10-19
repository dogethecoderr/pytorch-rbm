# Restricted Boltzmann Machines, Explained Through the Code

### For our project: *Quantum Machine Learning Based Adversarial Training for IoT Intrusion Detection*

Our proposal replaces the Gaussian noise prior of a GAN with samples from a **Boltzmann machine** trained on IoT attack traffic, sampled classically (Gibbs / contrastive divergence) or with **D-Wave quantum annealing**. This guide teaches what that Boltzmann machine actually is and how it's trained, by walking line by line through a working implementation:

- [rbm.py](rbm.py) — a Restricted Boltzmann Machine trained with contrastive divergence (94 lines)
- [mnist_example.py](mnist_example.py) — trains it on handwritten digits and tests what it learned

The MNIST digits are a stand-in. The model, the training algorithm and the failure modes are the same ones we'll face on Bot-IoT. Every section ends with a **For our project** note connecting the code to our pipeline, and Section 8 lays out the concrete path from this repo to the prior in our architecture.

You don't need to know anything about Boltzmann machines going in. Some comfort with Python and the idea of a matrix multiply is enough.

---

## Table of Contents

1. [Where this code fits in our pipeline](#1-where-this-code-fits-in-our-pipeline)
2. [The big idea in plain English](#2-the-big-idea-in-plain-english)
3. [The math, kept small](#3-the-math-kept-small)
4. [rbm.py, line by line](#4-rbmpy-line-by-line)
5. [mnist_example.py, line by line](#5-mnist_examplepy-line-by-line)
6. [What the results mean](#6-what-the-results-mean)
7. [Quirks to fix before this becomes our baseline](#7-quirks-to-fix-before-this-becomes-our-baseline)
8. [From this repo to our pipeline](#8-from-this-repo-to-our-pipeline)
9. [Common confusions to avoid](#9-common-confusions-to-avoid)
10. [Glossary](#10-glossary)

---

## 1. Where this code fits in our pipeline

Our architecture (proposal, Model Architecture figure) has eight boxes. This repo implements the core of box 3:

```
Our pipeline                                            This repo
──────────────────────────────────────────────          ──────────────────────────────────────
3. Class-conditional latent prior
   Energy model:  Classical RBM (baseline)       ◀───── rbm.py, the RBM class
                  Classical DBM (deeper)                (a stack of these)
                  QBM via D-Wave anneal          ◀───── same energy + a transverse field (§3.4)
   Sampling:      Gibbs / contrastive divergence ◀───── rbm.py, contrastive_divergence (lines 42–82)
                  Quantum annealing (Leap)       ◀───── replaces rbm.py lines 48–59 (§8.5)
4. Generator (PennyLane PQC)                            not in this repo
5. Adversarial training (cWGAN-GP)                      not in this repo
6–7. Augment training set → IDS classifier       ◀───── mnist_example.py lines 85–92 play this role
8. Evaluation (macro-F1, Wasserstein, ...)       ◀───── mnist_example.py prints accuracy (§6.6: why that isn't enough for us)
```

**One important difference in how the RBM is *used*:**

| | In mnist_example.py | In our pipeline |
|---|---|---|
| Role | **Feature extractor** (encoder) | **Generative prior** (sampler) |
| Input | A real image | Nothing (or a class label) |
| Output | 128 hidden probabilities describing that image | A random binary vector `z` drawn from what the RBM learned |
| Function used | `sample_hidden(real_data)` | A Gibbs chain or a quantum annealer run on the trained RBM |

The trained model is identical in both cases. What differs is whether we feed it data and read features out, or let it run on its own and collect what it generates. The training code (`contrastive_divergence`) is what we reuse directly.

---

## 2. The big idea in plain English

### Two layers, and that's it

An RBM is a neural network with exactly **two layers**:

- a **visible layer**: the data you can see (in this repo, the 784 pixels of a 28×28 image; for us, the binarized Bot-IoT flow features)
- a **hidden layer**: units that learn to detect patterns in the data (here, 128 of them)

```
 Hidden layer (128 units)     h1    h2    h3   ...  h128
                               │ ╲  ╱ │ ╲  ╱ │         │
                               │  ╳   │  ╳   │   ...   │      every visible unit connects
                               │ ╱  ╲ │ ╱  ╲ │         │      to every hidden unit
 Visible layer (784 pixels)   v1    v2    v3   ...  v784
```

Every visible unit is connected to every hidden unit, and each connection has a **weight**. That's 784 × 128 = **100,352 weights**, plus one **bias** per unit (784 + 128 = 912 biases), for **101,264 learnable numbers** total.

### What makes it "restricted"

In a general Boltzmann machine, the kind written in our proposal with a sum over all pairs `i < j`, any unit can connect to any other unit. That makes it very hard to train and sample, because units have to be updated one at a time.

The "restriction" is: **no connections within a layer**. That one rule means:

- if you know all the visible values, every hidden unit can be computed **independently and all at once**, and
- if you know all the hidden values, every visible unit can be computed **independently and all at once**.

In code, each of those "all at once" computations is a single matrix multiply. You'll see this in `sample_hidden` and `sample_visible`. For us, the restriction matters twice: it makes classical Gibbs sampling fast, and it gives a **bipartite** connectivity pattern, which is a structure the D-Wave hardware graph can embed more easily than an all-to-all one.

### What makes it a "Boltzmann" machine

A Boltzmann machine assigns an **energy** to every possible configuration of its units. Low energy means "this configuration is likely"; high energy means "unlikely." Probabilities come from energies through the Boltzmann distribution from statistical physics, which is exactly the distribution a quantum annealer is (approximately) able to sample from. That shared physics is why RBMs and D-Wave hardware fit together at all.

Training adjusts the weights so that **real data gets low energy** and **things that don't look like real data get high energy**.

### What it's actually learning

An RBM is a **generative**, **unsupervised** model:

- **Unsupervised:** it never needs labels. It learns the structure of the data itself.
- **Generative:** it learns a model of what the data looks like, well enough to produce new samples.

Each hidden unit ends up acting like a **feature detector**. On digits, that might be "a curved stroke in the upper left." On Bot-IoT traffic, it might be "high packet rate *together with* tiny payloads *together with* a short duration", which is a co-occurrence pattern typical of a flood attack.

### Why a Boltzmann prior instead of Gaussian noise

This is the core hypothesis of our proposal, so it's worth stating precisely.

- A standard GAN feeds its generator `z ~ N(0, I)`: every dimension independent, one smooth bell-shaped blob, **no information about the data**. The generator has to manufacture every bit of structure by itself.
- A sample from an RBM trained on attack traffic is a binary vector whose bits are **already correlated the way real attack features are correlated**, and whose distribution can have **many separate modes**, one cluster per attack pattern rather than one blob.

The bet (supported by Kim et al. [3] for Boltzmann vs. Gaussian priors) is that a generator starting from structured, multi-modal noise has an easier job producing realistic rare-class traffic. Our experiments test whether that holds for IoT intrusion data, and whether sampling the prior with quantum annealing helps further.

### The core training trick: "reality vs. dream"

Training compares two things:

1. **Reality (positive phase):** show the RBM a real example. Record which visible and hidden units are active *together*.
2. **Dream (negative phase):** let the RBM run on its own for a few steps, generating its own version. Record which units are active together *in the dream*.

Then nudge the weights:

- **strengthen** connections that are active together in reality,
- **weaken** connections that are active together in the dream.

When the dreams look just like reality, the two cancel and learning stops. This is **Contrastive Divergence (CD)**, the heart of [rbm.py](rbm.py).

**For our project:** the "dream" step is precisely where quantum annealing enters. Classically, the dream is a short Gibbs chain. In the quantum-annealing condition, the dream samples come from the D-Wave QPU instead (§4, negative phase; §8.5).

---

## 3. The math, kept small

You can skip this section and still follow the code. But every formula maps to a specific line, and §3.3–3.4 connect directly to the equations in our proposal.

### 3.1 Notation

| Symbol | Meaning | In the code |
|---|---|---|
| `v` | visible values, each 0 or 1 | `input_data`, `visible_probabilities` |
| `h` | hidden values, each 0 or 1 | `hidden_activations` |
| `W` | weight matrix, shape 784 × 128 | `self.weights` |
| `a` | visible biases (784 numbers) | `self.visible_bias` |
| `b` | hidden biases (128 numbers) | `self.hidden_bias` |
| `σ(x)` | sigmoid: `1 / (1 + e^-x)`, squashes any number into 0 to 1 | `self._sigmoid` |

### 3.2 Energy, probability, and the learning rule

**Energy** of one configuration:

```
E(v, h) = − Σᵢ aᵢvᵢ − Σⱼ bⱼhⱼ − Σᵢⱼ vᵢ Wᵢⱼ hⱼ
```

Energy goes *down* (more likely) when a unit with a positive bias is on, and when two connected units with a positive weight are both on.

**Probability** of a configuration:

```
P(v, h) = e^(−E(v, h)) / Z
```

`Z` sums `e^(−E)` over every possible configuration. With 128 binary hidden units alone there are 2¹²⁸ hidden configurations, so `Z` can't be computed directly. That's why training needs a shortcut.

**The two formulas the code actually uses.** Because there are no within-layer connections:

```
P(hⱼ = 1 | v) = σ( bⱼ + Σᵢ vᵢ Wᵢⱼ )        → sample_hidden,  rbm.py:33-34
P(vᵢ = 1 | h) = σ( aᵢ + Σⱼ Wᵢⱼ hⱼ )        → sample_visible, rbm.py:38-39
```

In matrix form for a whole batch: `σ(v @ W + b)` and `σ(h @ Wᵀ + a)`.

**The learning rule.** The gradient of the log-probability of the data with respect to a weight is:

```
∂ log P(v) / ∂Wᵢⱼ  =  ⟨vᵢ hⱼ⟩_data  −  ⟨vᵢ hⱼ⟩_model
                       ─────────────     ──────────────
                       "reality"          "dream"
                       positive phase     negative phase
```

`⟨vᵢ hⱼ⟩` means "how often unit `i` and unit `j` are on together, on average." The data term is easy. The model term needs samples from the model's own distribution. **CD-k** approximates it by starting at real data and running only `k` back-and-forth Gibbs steps. **Quantum-annealing-assisted training** approximates it with samples from the QPU instead.

The bias rules are the same idea with one side dropped:

```
∂/∂aᵢ = ⟨vᵢ⟩_data − ⟨vᵢ⟩_model           → rbm.py:66
∂/∂bⱼ = ⟨hⱼ⟩_data − ⟨hⱼ⟩_model           → rbm.py:69
```

### 3.3 From 0/1 units (this code) to ±1 spins (our proposal and D-Wave)

Our proposal writes units as spins `s ∈ {−1, +1}` with energy `E(s) = −Σ bᵢsᵢ − Σ wᵢⱼsᵢsⱼ`, which is the **Ising** convention used in physics and by D-Wave. This code uses `v, h ∈ {0, 1}`, the **QUBO / binary** convention used in machine learning. They describe the same model; only the bookkeeping differs.

Substituting `v = (1 + s)/2` for visible units and `h = (1 + t)/2` for hidden units into the energy in §3.2 gives (up to a constant, which doesn't change any probability):

```
Ising field on visible unit i:   aᵢ/2 + ¼ Σⱼ Wᵢⱼ
Ising field on hidden unit j:    bⱼ/2 + ¼ Σᵢ Wᵢⱼ
Ising coupling between i and j:  ¼ Wᵢⱼ
```

Two practical consequences:

- **The numbers change.** A weight of 0.4 in this code is a coupling of 0.1 in the Ising form, and every bias picks up a contribution from its row or column of weights. If the classical and quantum conditions are meant to use "the same RBM," this conversion has to be applied exactly.
- **You often don't need to do it by hand.** D-Wave's `dimod` library accepts models in either convention (`dimod.BINARY` or `dimod.SPIN`) and converts for you (§8.5). The formulas above are still worth knowing so you can sanity-check what gets sent to the hardware.

### 3.4 How this relates to the Quantum Boltzmann Machine (QBM)

Our proposal's QBM uses the transverse-field Ising Hamiltonian:

```
H = − Σᵢ Γᵢ σᵢˣ − Σᵢ bᵢ σᵢᶻ − Σᵢ<ⱼ wᵢⱼ σᵢᶻ σⱼᶻ
```

Set every `Γᵢ = 0` and the first term disappears. The remaining operators only have the ordinary on/off values ±1, and `H` becomes the classical Ising energy from §3.3. So:

> **The RBM in rbm.py is the Γ = 0 special case of our QBM** (restricted to bipartite wiring and written in 0/1 form).

Everything in this guide about energies, positive and negative phases, and biases carries over. What the transverse field adds is quantum fluctuation between on and off, and a density matrix in place of a probability distribution. It also makes training harder: the gradient for the `Γ` terms can't be estimated from samples the way §3.2's gradient can, which is why published QBM training (Amin et al., 2018) optimizes a bound on the likelihood instead. Keep the two quantum conditions distinct:

| Condition | Energy model | Sampler |
|---|---|---|
| Classical RBM | rbm.py (Γ = 0) | Gibbs / CD-k (rbm.py) |
| RBM sampled by quantum annealing | rbm.py (Γ = 0) | D-Wave QPU (Dixit et al. [4]) |
| QBM | Γ ≠ 0 | D-Wave QPU, with QBM-specific training |

---

## 4. rbm.py, line by line

### Line 1: the only import

```python
import torch
```

The RBM is written with raw PyTorch tensors. It does **not** use `torch.nn` or autograd. The RBM's learning rule (§3.2) is computed by hand, so there's no loss function to call `.backward()` on.

> **For our project:** this matters when we connect the RBM to the WGAN-GP. The generator and critic will use autograd as usual; the RBM prior sits outside that graph and is trained by its own rule. If we later want the GAN's gradients to also update the prior (the "gradients update generator + energy model" arrow in our figure), that coupling has to be designed explicitly, because the RBM has no `.backward()`.

### Line 4: the class

```python
class RBM():
```

A plain Python class, not a `torch.nn.Module`. It holds tensors and updates them manually.

### Lines 6–13: the constructor and hyperparameters

```python
def __init__(self, num_visible, num_hidden, k, learning_rate=1e-3,
             momentum_coefficient=0.5, weight_decay=1e-4, use_cuda=True):
```

| Parameter | Value in the MNIST example | What it controls |
|---|---|---|
| `num_visible` | 784 | Size of the visible layer: one unit per pixel |
| `num_hidden` | 128 | Size of the hidden layer: how many feature detectors to learn |
| `k` | 2 | Number of Gibbs steps in the "dream" (the *k* in CD-*k*) |
| `learning_rate` | 0.001 | How big each weight update is |
| `momentum_coefficient` | 0.5 | How much of the previous update carries over into the next one |
| `weight_decay` | 0.0001 | How strongly weights are pulled toward zero each step |
| `use_cuda` | `False` on a Mac | Whether to run on an NVIDIA GPU |

Lines 7–13 just store these on `self`.

> **For our project:** `num_visible` becomes the number of bits after binarizing the Bot-IoT features (§8.1), likely tens rather than hundreds. `num_hidden` is bounded by two hardware limits: the QPU embedding (§8.7) and, if hidden samples become the latent `z`, the number of qubits the PennyLane generator can take (§8.6). All of these hyperparameters must be identical across the classical and quantum conditions, or the comparison isn't fair.

### Lines 15–17: initializing the parameters

```python
self.weights = torch.randn(num_visible, num_hidden) * 0.1
self.visible_bias = torch.ones(num_visible) * 0.5
self.hidden_bias = torch.zeros(num_hidden)
```

- **Weights** start as small random numbers (normal distribution, standard deviation 0.1). They must be random: if every weight started equal, every hidden unit would compute the same thing and learn the same feature forever. They must be *small* so the sigmoid starts in its sensitive middle range instead of saturated at 0 or 1.
- **Visible biases** start at 0.5: a mild initial push toward "on."
- **Hidden biases** start at 0: no initial preference.

Shapes: `weights` is `[784, 128]`, so row `i` holds all connections out of visible unit `i`, and column `j` holds all connections into hidden unit `j`.

> **For our project:** Hinton's practical guide recommends initializing each visible bias to `log(pᵢ / (1 − pᵢ))`, where `pᵢ` is the fraction of training examples in which unit `i` is on. For heavily skewed binary features (a bit that's on in 0.1% of flows), that's very different from 0.5 and speeds up training noticeably.

### Lines 19–21: momentum buffers

```python
self.weights_momentum = torch.zeros(num_visible, num_hidden)
self.visible_bias_momentum = torch.zeros(num_visible)
self.hidden_bias_momentum = torch.zeros(num_hidden)
```

One "velocity" tensor per parameter, same shape, starting at zero. Explained in the update step (lines 62–69).

### Lines 23–30: moving to the GPU

```python
if self.use_cuda:
    self.weights = self.weights.cuda()
    ...
```

If an NVIDIA GPU is available, every tensor is copied to GPU memory. On a Mac this is skipped and everything runs on the CPU.

### Lines 32–35: `sample_hidden` — "what patterns do I see?"

```python
def sample_hidden(self, visible_probabilities):
    hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
    hidden_probabilities = self._sigmoid(hidden_activations)
    return hidden_probabilities
```

This is `P(h = 1 | v) = σ(v @ W + b)`. For a batch of 64 examples:

```
visible_probabilities   [64, 784]
        @ weights       [784, 128]
                      = [64, 128]   ← one score per example, per hidden unit
        + hidden_bias   [128]       ← broadcast to every row
        sigmoid       → [64, 128]   ← probability each hidden unit is on
```

**Intuition:** each hidden unit `j` takes a weighted vote of all visible units. Units with a large positive weight to `j` push it on; units with a negative weight push it off. The column of weights for one hidden unit is a "template," and the unit fires when the input matches it.

**Naming note:** despite the name, this function returns *probabilities*, not samples. The actual 0/1 sampling happens separately on lines 45 and 54.

> **For our project:** a GAN prior needs real binary samples, not probabilities. Whenever we draw `z` from the RBM, wrap the result: `torch.bernoulli(rbm.sample_hidden(v))`.

### Lines 37–40: `sample_visible` — "what input would produce these patterns?"

```python
def sample_visible(self, hidden_probabilities):
    visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias
    visible_probabilities = self._sigmoid(visible_activations)
    return visible_probabilities
```

The mirror image of `sample_hidden`: `P(v = 1 | h) = σ(h @ Wᵀ + a)`.

```
hidden_probabilities    [64, 128]
        @ weights.t()   [128, 784]   ← same weights, transposed
                      = [64, 784]
        + visible_bias, sigmoid → [64, 784]   ← a reconstructed input
```

**Key idea:** the **same weight matrix** is used in both directions. An RBM has no separate encoder and decoder. Information flows up through `W` and back down through `Wᵀ`. That symmetry is what makes it a single probabilistic model, and why it maps onto a single set of couplers on the annealer.

### Lines 42–82: `contrastive_divergence` — one training step

This is the whole learning algorithm, called once per batch.

#### Positive phase (lines 43–46): reality

```python
positive_hidden_probabilities = self.sample_hidden(input_data)
positive_hidden_activations = (positive_hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()
positive_associations = torch.matmul(input_data.t(), positive_hidden_activations)
```

1. **Line 44:** show real data to the hidden layer and get the probability that each hidden unit turns on.
2. **Line 45:** turn those probabilities into on/off decisions. Draw a random number `r` between 0 and 1, and switch the unit on if `probability >= r`. A unit with probability 0.8 passes 80% of the time. `.float()` converts `True/False` to `1.0/0.0`.
3. **Line 46:** compute `vᵀh`, shape `[784, 64] @ [64, 128] = [784, 128]`. Entry `(i, j)` counts **how often visible unit `i` and hidden unit `j` were on together in real data**: the `⟨vᵢhⱼ⟩_data` term.

**Why sample to 0/1 instead of using the probability?** Binary samples act as an information bottleneck and add noise that regularizes training. A hidden unit can't pass along "0.73"; it has to commit.

#### Negative phase (lines 48–59): the dream

```python
hidden_activations = positive_hidden_activations

for step in range(self.k):
    visible_probabilities = self.sample_visible(hidden_activations)
    hidden_probabilities = self.sample_hidden(visible_probabilities)
    hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()
```

Starting from the hidden pattern produced by real data, the RBM bounces back and forth `k = 2` times:

```
real data ──▶ h₀ ──▶ v₁ ──▶ h₁ ──▶ v₂ ──▶ h₂
         (positive)  └─ step 1 ─┘  └─ step 2 ─┘
```

Each round trip is one **Gibbs sampling** step. With each step, the sample drifts away from the real input toward what the model "believes" the data looks like.

```python
negative_visible_probabilities = visible_probabilities
negative_hidden_probabilities = hidden_probabilities

negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)
```

- **Lines 56–57:** keep the final dream sample and hidden probabilities.
- **Line 59:** the same `vᵀh` calculation as line 46, but on the dream: `⟨vᵢhⱼ⟩_model`.

The negative phase uses **probabilities** for the final values. That's a standard trick from Hinton's practical guide: a less noisy estimate of the same average.

> **For our project — this block is where quantum annealing plugs in.**
>
> The true `⟨vᵢhⱼ⟩_model` requires samples from the model's full distribution, which would take a Gibbs chain running for a very long time. CD-k cheats in two ways: it runs only `k` steps, and it **starts at real data**. So its "dream" never strays far from the training examples, and the gradient is **biased**. In practice, CD-trained RBMs can assign high probability to spurious configurations that the short chains never visit.
>
> The quantum-annealing condition replaces lines 48–59 with samples drawn by the D-Wave QPU from the model's energy function directly, not started from data (Adachi & Henderson, 2015; Dixit et al. [4]). If those samples are closer to the true model distribution than CD-k's, the gradient is less biased and the learned prior should be more faithful. That's the mechanism behind our claim, and §8.5 sketches the code.

#### Parameter update (lines 61–77)

```python
self.weights_momentum *= self.momentum_coefficient
self.weights_momentum += (positive_associations - negative_associations)
```

**Line 63 is the learning rule.** `positive_associations − negative_associations` is "reality minus dream":

| Situation | Positive | Negative | Update | Effect |
|---|---|---|---|---|
| Units `i` and `j` on together in real data, but not in dreams | high | low | **+** | strengthen the connection so dreams look more real |
| On together in dreams, but not in real data | low | high | **−** | weaken it, so the model stops imagining things that aren't there |
| Dreams already match reality | ≈ | ≈ | **≈ 0** | nothing left to learn |

**Momentum (line 62):** before adding the new gradient, the old velocity is multiplied by 0.5. Each update is "this batch's gradient + half the last update + a quarter of the one before ...". This smooths noise between batches. With a coefficient of 0.5, a steady direction gets up to `1 / (1 − 0.5) = 2×` the step size.

```python
self.visible_bias_momentum *= self.momentum_coefficient
self.visible_bias_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

self.hidden_bias_momentum *= self.momentum_coefficient
self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)
```

The same "reality minus dream" logic for the biases:

- **Line 66:** if a visible unit is on more in real data than in dreams, raise its bias.
- **Line 69:** if a hidden unit is on more for real data than in dreams, raise its bias.

`dim=0` sums over the examples in the batch.

```python
batch_size = input_data.size(0)

self.weights += self.weights_momentum * self.learning_rate / batch_size
self.visible_bias += self.visible_bias_momentum * self.learning_rate / batch_size
self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / batch_size
```

**Lines 73–75 apply the update.**

- **`+=`, not `−=`.** Most networks *minimize a loss*, so they subtract the gradient. The RBM *maximizes the probability of the data*, so it adds: gradient **ascent**.
- **`/ batch_size`** turns sums into averages, so the step size doesn't depend on batch size. This matters for the last batch, which is smaller.

```python
self.weights -= self.weights * self.weight_decay  # L2 weight decay
```

**Line 77: weight decay.** Every step, each weight shrinks by 0.01%. This is L2 regularization: it keeps weights from growing large, which limits overfitting and keeps sigmoids out of saturation. Over one MNIST epoch (938 batches), a weight with no gradient pushing on it would shrink to about `0.9999^938 ≈ 91%` of its value; over 10 epochs, to about 39%. Biases are not decayed.

> **For our project:** keeping weights small has a second benefit on the annealer. The QPU has a limited range of allowed couplings and fields, and large weights force more aggressive rescaling, which costs precision (§8.5).

#### Reconstruction error (lines 79–82)

```python
error = torch.sum((input_data - negative_visible_probabilities)**2)
return error
```

The squared difference between the real data and the dream (after `k = 2` steps), summed over every unit in the batch.

**Important:** this is **not** what the RBM optimizes. It optimizes the probability of the data, which can't be computed because of `Z`. Reconstruction error is a cheap **progress indicator**.

> **For our project:** Kim et al. [3] use reconstruction error to compare Boltzmann and Gaussian priors, so we may report it too. Two cautions: (1) the number depends on `k` and on whether probabilities or samples are compared, so the definition must be fixed and identical across all conditions; (2) low reconstruction error doesn't prove the RBM generates good samples. Our synthetic-data checks (Wasserstein distance, correlation structure, nearest-neighbor memorization) should be applied to the RBM's own samples as well, not only to the final GAN output.

### Lines 84–85: `_sigmoid`

```python
def _sigmoid(self, x):
    return 1 / (1 + torch.exp(-x))
```

The logistic function. It maps any number to the range 0 to 1: large positive → near 1, large negative → near 0, zero → exactly 0.5. (`torch.sigmoid` does the same thing and is more numerically stable.)

### Lines 87–93: `_random_probabilities`

```python
def _random_probabilities(self, num):
    random_probabilities = torch.rand(num)
    if self.use_cuda:
        random_probabilities = random_probabilities.cuda()
    return random_probabilities
```

Returns `num` uniform random numbers between 0 and 1, used as the coin-flip thresholds on lines 45 and 54. Its shape causes a real problem for us; see §7, item 1.

---

## 5. mnist_example.py, line by line

This file answers: **did the RBM learn anything useful?** It trains an RBM on handwritten digits *without labels*, then checks whether its hidden features make digits easy to classify. That's structurally the same question as our augmentation experiment: does something learned without supervision help a downstream classifier?

### Lines 1–8: imports

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.datasets
import torchvision.models
import torchvision.transforms

from rbm import RBM
```

| Import | Used for |
|---|---|
| `numpy` | Arrays to hold the extracted features for scikit-learn |
| `LogisticRegression` | A simple classifier, used to test the features |
| `torch` | Tensors, data loading, CUDA detection |
| `torchvision.datasets` | Downloads MNIST |
| `torchvision.models` | **Not used.** Leftover import |
| `torchvision.transforms` | Converts images to tensors |
| `RBM` | The class from rbm.py |

### Lines 11–24: configuration

```python
BATCH_SIZE = 64
VISIBLE_UNITS = 784  # 28 x 28 images
HIDDEN_UNITS = 128
CD_K = 2
EPOCHS = 10

DATA_FOLDER = 'data/mnist'

CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0

if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)
```

- **`BATCH_SIZE = 64`**: 60,000 / 64 → **938 updates per epoch** (the last batch has 32).
- **`VISIBLE_UNITS = 784`**: fixed by the data, 28 × 28 pixels.
- **`HIDDEN_UNITS = 128`**: a design choice. Fewer hidden units than visible forces the RBM to find structure rather than memorize.
- **`CD_K = 2`**: two dream steps per update.
- **`EPOCHS = 10`**: **9,380 updates** total.
- **`CUDA`**: `True` only if an NVIDIA GPU is available.

### Lines 27–34: loading MNIST

```python
train_dataset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=True,
    transform=torchvision.transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

test_dataset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=False,
    transform=torchvision.transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
```

**MNIST** is 70,000 grayscale digit images: 60,000 for training, 10,000 for testing. `ToTensor()` scales pixels from 0–255 to 0.0–1.0, which the RBM treats as "probability this unit is on." MNIST is nearly black-and-white, so it fits a binary RBM well. `DataLoader` batches the images but does **not** shuffle them.

> **For our project:** Bot-IoT is nothing like this. Its features (byte counts, packet rates, durations, protocol) are continuous, heavy-tailed and on wildly different scales, and the QPU accepts only strictly binary units. This loading step becomes our biggest preprocessing job (§8.1). Also, **shuffle**: network-traffic CSVs are often grouped by attack type or capture time, so unshuffled batches could each contain a single class, which badly distorts CD training.

### Lines 37–55: training the RBM

```python
rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, use_cuda=CUDA)

for epoch in range(EPOCHS):
    epoch_error = 0.0

    for batch, _ in train_loader:
        batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data

        if CUDA:
            batch = batch.cuda()

        batch_error = rbm.contrastive_divergence(batch)

        epoch_error += batch_error

    print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))
```

- **Line 40:** create the RBM; everything not passed uses defaults.
- **Line 45:** `for batch, _ in train_loader`. The `_` discards labels: **the RBM never sees which digit an image is.**
- **Line 46:** flatten each image from `[1, 28, 28]` to a row of 784 numbers. The RBM doesn't know pixels form a grid; any spatial structure it learns, it discovers on its own. (Bot-IoT rows are already flat, so this step disappears.)
- **Line 51:** one CD-2 training step.
- **Lines 53–55:** sum the reconstruction error and print it once per epoch.

> **For our project:** our prior is *class-conditional* (box 3: "Conditioned latent vector, tagged by class"), so we can't throw the labels away. §8.2 covers the two standard ways to bring them in.

### Lines 58–82: extracting features

```python
train_features = np.zeros((len(train_dataset), HIDDEN_UNITS))   # [60000, 128]
train_labels = np.zeros(len(train_dataset))                     # [60000]
test_features = np.zeros((len(test_dataset), HIDDEN_UNITS))     # [10000, 128]
test_labels = np.zeros(len(test_dataset))                       # [10000]

for i, (batch, labels) in enumerate(train_loader):
    batch = batch.view(len(batch), VISIBLE_UNITS)
    ...
    train_features[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = rbm.sample_hidden(batch).cpu().numpy()
    train_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()
```

Each image goes through `sample_hidden` **once**, producing 128 hidden probabilities: the image's **features**. The slice `i*BATCH_SIZE : i*BATCH_SIZE+len(batch)` places batch `i` in the right rows, and `len(batch)` handles the short final batch. Lines 75–82 do the same for the test set.

Each image went from **784 raw values** to **128 learned features**, using a mapping learned without labels.

> **For our project:** this is the *encoder* use of an RBM, which our pipeline doesn't need (see the table in §1). It's still a useful diagnostic: if hidden features of a Bot-IoT RBM separate attack classes well under a simple classifier, the RBM has learned structure that distinguishes those classes, which is the property we want the prior to carry into the generator.

### Lines 85–92: classification

```python
clf = LogisticRegression()
clf.fit(train_features, train_labels)
predictions = clf.predict(test_features)

print('Result: %d/%d' % (sum(predictions == test_labels), test_labels.shape[0]))
```

A logistic regression, a simple linear classifier, is trained on the 60,000 training images' RBM features plus their labels, then labels the 10,000 test images. If the features captured what makes a 3 different from an 8, even a linear classifier does well.

**The overall pipeline:**

```
                    (no labels)                         (labels used here only)
60,000 images ──▶ train RBM with CD-2 ──▶ RBM weights
                                              │
60,000 images ──▶ sample_hidden ──────────────┴──▶ 128 features each ──▶ fit logistic regression
10,000 images ──▶ sample_hidden ─────────────────▶ 128 features each ──▶ predict ──▶ accuracy
```

> **For our project:** the classifier slot is where LightGBM, Random Forest, XGBoost and MLP go (box 7), and the input changes from "RBM features of real data" to "real data + GAN-generated rare-class samples." The final `print` is the slot for our evaluation, which must be macro-F1 and PR-AUC, not accuracy (§6.6).

---

## 6. What the results mean

Here's the actual output of running `python mnist_example.py` on a Mac (CPU only), with download progress bars trimmed:

```
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Failed to download (trying next):
HTTP Error 404: Not Found
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
...
Training RBM...
Epoch Error (epoch=0): 5816332.5000
Epoch Error (epoch=1): 3121309.0000
Epoch Error (epoch=2): 2455372.0000
Epoch Error (epoch=3): 2222712.0000
Epoch Error (epoch=4): 2103535.5000
Epoch Error (epoch=5): 2022649.5000
Epoch Error (epoch=6): 1962220.1250
Epoch Error (epoch=7): 1917456.7500
Epoch Error (epoch=8): 1883702.3750
Epoch Error (epoch=9): 1854012.2500
Extracting features...
Classifying...
ConvergenceWarning: lbfgs failed to converge after 100 iteration(s) ...
Result: 9305/10000
```

### 6.1 The download messages

The `HTTP Error 404` lines are harmless. MNIST's original host no longer serves the files, so torchvision fell back to a mirror on Amazon S3. The data is saved in `data/mnist/` and later runs skip the download.

### 6.2 The epoch errors: did the RBM learn?

The raw numbers are summed over **60,000 images × 784 pixels ≈ 47 million comparisons** per epoch. Dividing them out:

| Epoch | Total error | Per image | Per pixel | Drop from previous epoch |
|---|---|---|---|---|
| 0 | 5,816,332 | 96.9 | 0.124 | — |
| 1 | 3,121,309 | 52.0 | 0.066 | −46% |
| 2 | 2,455,372 | 40.9 | 0.052 | −21% |
| 3 | 2,222,712 | 37.0 | 0.047 | −9.5% |
| 4 | 2,103,536 | 35.1 | 0.045 | −5.4% |
| 5 | 2,022,650 | 33.7 | 0.043 | −3.8% |
| 6 | 1,962,220 | 32.7 | 0.042 | −3.0% |
| 7 | 1,917,457 | 32.0 | 0.041 | −2.3% |
| 8 | 1,883,702 | 31.4 | 0.040 | −1.8% |
| 9 | 1,854,012 | 30.9 | 0.039 | −1.6% |

1. **The RBM clearly learned.** Error fell 68% overall.
2. **Typical learning-curve shape.** Big gains early (−46% in epoch 1), then steadily smaller ones. Coarse structure is learned fast; fine detail slowly.
3. **Not fully converged.** Still falling 1.6% per epoch at the end.
4. **"0.039 per pixel"** is average squared error; its square root, about **0.20**, is the typical brightness difference between a real pixel and its dreamed version on a 0-to-1 scale. It's most likely concentrated on stroke edges, while the empty background is easy to reconstruct.
5. **Epoch 0 isn't the starting error.** It's accumulated *during* the first epoch while the model was already learning.

### 6.3 The ConvergenceWarning

This comes from scikit-learn's logistic regression, **not** the RBM. Its L-BFGS optimizer stops after 100 iterations by default, which wasn't quite enough to settle. `LogisticRegression(max_iter=1000)` silences it.

### 6.4 The final result: 9305 / 10000

**93.05% of test digits were classified correctly** (695 wrong).

| Comparison | Accuracy | What it tells you |
|---|---|---|
| Random guessing | 10% | Ten classes |
| Logistic regression on **raw pixels** (784 features) | around 92% (typical published figure; not measured in this run) | A simple classifier without the RBM |
| **Logistic regression on RBM features (this run)** | **93.05%** | 128 features learned without labels |
| README's reported result | 92.8% | Same code, different random run |
| Modern convolutional networks | 99.5%+ | State of the art |

> **The RBM compressed each image from 784 numbers to 128, without seeing a label, and the compressed version is at least as useful for recognizing digits as the full image.**

That's evidence the RBM captured the real structure of the data, which is the same property our prior needs.

### 6.5 Why 93.05% and not 92.8%?

The code never sets a random seed. The initial weights and every coin flip during training differ on each run, so results move by a few tenths of a percent from luck alone.

### 6.6 What these results teach us for our project

**1. Accuracy would be misleading on Bot-IoT.** MNIST is balanced (about 10% of examples per digit), so accuracy is a fair summary there. Bot-IoT is the opposite: DDoS and DoS make up roughly 97% of records, while Theft has 1,587 of about 73 million. A classifier that always answers "DDoS or DoS" scores around 97% accuracy while detecting **zero** theft attacks. That's why our evaluation uses **macro-F1** (each class counts equally) and **PR-AUC** (not fooled by a huge majority class), plus per-class precision and recall for the rare classes.

**2. Reconstruction error falling isn't enough evidence.** It showed this RBM was learning, but it doesn't show that the RBM's *samples* look like real attacks. For the prior, we should run the same three synthetic-data checks from our proposal on raw RBM samples (decoded back to feature space): per-feature Wasserstein distance, correlation structure, and nearest-neighbor distance for memorization. The correlation check matters most, because co-occurrence structure is exactly what the Boltzmann prior is supposed to add over Gaussian noise.

**3. Run-to-run noise is as large as some effects we might measure.** Changing nothing but the random seed moved accuracy by 0.25 percentage points here. If the quantum condition beats the classical one by a similar margin, that could be pure luck. For the "Statistical analysis" section, every condition should be run with several seeds and reported as mean ± standard deviation, with a significance test across conditions. QPU sampling adds hardware noise on top of seed noise, which makes repeated runs even more important.

**4. Cost is worth tracking from day one.** This 100k-parameter RBM trained in a few minutes on a laptop CPU. A Bot-IoT RBM will be far smaller in parameters, so classical training is cheap. QPU time is the scarce resource, which is why our evaluation box lists QPU-seconds vs. wall-clock. Logging both for every run makes that comparison possible later.

---

## 7. Quirks to fix before this becomes our baseline

This code works (93% on MNIST), but our classical-RBM condition is the baseline the quantum conditions are judged against. **If the baseline is handicapped by implementation quirks, any quantum advantage we report is inflated**, and a reviewer comparing against a correct RBM would catch it. In priority order:

1. **Random thresholds are shared across the batch** ([rbm.py:45](rbm.py#L45), [rbm.py:54](rbm.py#L54)). `_random_probabilities(self.num_hidden)` returns shape `[128]`, which PyTorch broadcasts across all rows of the `[64, 128]` batch. **Every example in a batch uses the same random draw for each hidden unit**, so samples within a batch are correlated. That adds noise to training and, for a prior, directly reduces sample diversity. **Fix:** replace both lines with `torch.bernoulli(...)` on the probabilities.

2. **No shuffling** ([mnist_example.py:31](mnist_example.py#L31)). Harmless-ish for MNIST, risky for traffic data that's ordered by class or time. **Fix:** `shuffle=True`.

3. **No random seed.** Results aren't reproducible (§6.5). **Fix:** `torch.manual_seed(seed)` and `numpy.random.seed(seed)`, with the seed logged for every run.

4. **`sample_hidden` / `sample_visible` don't sample.** They return probabilities. **Fix:** wrap with `torch.bernoulli` whenever binary samples are needed, or add explicit sampling methods.

5. **Weight decay isn't scaled by the learning rate** ([rbm.py:77](rbm.py#L77)). Many RBM implementations use `learning_rate × weight_decay × W`; this one applies `weight_decay × W` directly, about 1000× stronger for the same number. Not wrong, but hyperparameters copied from papers won't mean the same thing here. **Fix:** document it, and use identical settings across all conditions.

6. **Grayscale inputs in a binary model.** Fine classically (each value is treated as a probability). Not possible on the QPU, which only takes binary units, so for comparability **all conditions should train on the same strictly binary data** (§8.1).

7. **Positive phase uses 0/1 hidden samples; negative phase uses probabilities** ([rbm.py:46](rbm.py#L46) vs [rbm.py:59](rbm.py#L59)). A valid design choice, not a bug, but it must be kept the same when the negative phase is swapped for QPU samples, or the two conditions differ in more than the sampler.

8. **Unused import** `torchvision.models` ([mnist_example.py:5](mnist_example.py#L5)).

---

## 8. From this repo to our pipeline

A step-by-step path from the code above to box 3 of our architecture. The code below is a sketch to show the shape of each step; none of it has been run yet.

### 8.1 Turn Bot-IoT features into binary visible units

The QPU needs strictly binary units, and §7 item 6 says all conditions should share the same data, so binarize once and use it everywhere.

| Feature type | Example | Encoding |
|---|---|---|
| Categorical | protocol, state | One-hot: one visible unit per category |
| Heavy-tailed numeric | bytes, packets, rate | `log1p`, then split into quantile bins; one-hot the bin (or thermometer code: unit `k` on if value ≥ bin edge `k`) |
| Already binary | flags | One unit, as-is |

With the ten-feature Bot-IoT subset and around 4 bins per numeric feature, that's on the order of **40 visible units**: tiny compared with MNIST's 784, which is good for the QPU.

Two rules: fit bin edges on the **training split only** (otherwise test information leaks in), and keep the encoder so generated samples can be **decoded** back into feature values for the Wasserstein and correlation checks.

### 8.2 Make the prior class-conditional

Two standard options:

- **Label units (one RBM).** Append a one-hot class label to the visible layer, so each training row is `[features | one-hot class]`. To generate a class, **clamp** the label units to that class and sample everything else. Classically: reset the label units after every Gibbs step. On D-Wave: fix those variables in the model before sending it (`dimod`'s `fix_variable`). Rare classes share the weights learned from common classes, which helps when a class has only ~1,600 examples.
- **One RBM per class.** Simpler, but an RBM trained on 1,587 Theft rows alone may overfit or memorize, which our nearest-neighbor check would flag.

The label-unit approach is the better default for rare classes.

### 8.3 Train the classical baseline

Use `rbm.py` with the §7 fixes. The key change, on lines 45 and 54:

```python
positive_hidden_activations = torch.bernoulli(positive_hidden_probabilities)
...
hidden_activations = torch.bernoulli(hidden_probabilities)
```

Plus `shuffle=True` in the `DataLoader` and a logged seed. This trained model is the **"Classical RBM + Gibbs/CD"** condition.

### 8.4 Draw prior samples classically

After training, generate `z` by running a Gibbs chain from random noise, long enough to forget its starting point ("burn-in"):

```python
def sample_prior(rbm, n, steps=1000):
    v = torch.bernoulli(torch.full((n, rbm.num_visible), 0.5))
    for _ in range(steps):
        h = torch.bernoulli(rbm.sample_hidden(v))
        v = torch.bernoulli(rbm.sample_visible(h))
        # for a class-conditional prior, re-clamp the label units of v here
    return v, h
```

Whether `z` is the hidden state `h`, the visible state `v`, or both is a design decision; write it down and use the same choice in every condition. Long Gibbs chains can still get stuck in one mode of a multi-modal distribution. That slow mixing is part of the motivation for trying annealing.

### 8.5 Swap in quantum annealing for the negative phase

The idea: at each training step, build the RBM's current energy function as a model D-Wave understands, draw samples from the QPU, and use them in place of lines 48–59.

```python
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite

def rbm_to_bqm(rbm, beta_eff):
    W = rbm.weights.cpu().numpy()
    a = rbm.visible_bias.cpu().numpy()
    b = rbm.hidden_bias.cpu().numpy()
    linear = {f"v{i}": -a[i] / beta_eff for i in range(rbm.num_visible)}
    linear.update({f"h{j}": -b[j] / beta_eff for j in range(rbm.num_hidden)})
    quadratic = {(f"v{i}", f"h{j}"): -W[i, j] / beta_eff
                 for i in range(rbm.num_visible) for j in range(rbm.num_hidden)}
    return dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)

sampler = EmbeddingComposite(DWaveSampler())
sampleset = sampler.sample(rbm_to_bqm(rbm, beta_eff), num_reads=100)
# convert sampleset to tensors v_model [100, num_visible], h_model [100, num_hidden]
# negative_associations = v_model.t() @ h_model * (batch_size / 100)   ← match the positive phase's scale
```

The parts that make this a research problem rather than a drop-in:

- **`dimod.BINARY` means no manual ±1 conversion.** The energy is written exactly as in §3.2 (signs flipped because D-Wave adds the energy terms rather than subtracting them); `dimod` handles the Ising conversion from §3.3.
- **Effective temperature (`beta_eff`).** The QPU doesn't sample `exp(−E)`; at best it approximates `exp(−β_eff · E_sent)` for some unknown `β_eff` that depends on the hardware and problem scale. To target the RBM's distribution, `β_eff` has to be estimated and divided out (Benedetti et al., 2016). Getting it wrong means sampling a distribution that's too sharp or too flat.
- **Scaling and precision.** Hardware fields and couplings have limited ranges. By default the sampler rescales the problem to fit, which itself changes the effective temperature. Keep the scaling choice fixed when estimating `β_eff`.
- **Embedding and chains.** Each logical unit is usually represented by a "chain" of several physical qubits, because the hardware graph isn't fully bipartite. Broken chains corrupt samples, and chain strength is another setting to tune and report.
- **Develop without QPU time first.** `dwave-samplers` provides a classical simulated-annealing sampler with the same interface. Get the whole loop working with it, then switch to `DWaveSampler`.
- **Matched scale.** QPU samples aren't paired with batch rows, so rescale `negative_associations` to match the positive phase's batch sum. The same applies to the bias terms on lines 66 and 69.

This trained model is the **"RBM + quantum annealing"** condition. A QBM (Γ ≠ 0) is a further step with its own training method (§3.4).

### 8.6 Hand samples to the generator

The generator receives `z` instead of Gaussian noise. With a PennyLane PQC, a binary `z` maps naturally to **angle encoding**: rotate qubit `k` by π if bit `k` is on, which prepares a computational basis state. Simulated PQCs get expensive quickly as qubits are added, so the latent size (and therefore `num_hidden`, if `z = h`) has to stay small. A latent of a few to ~16 bits is realistic for a simulator; the proposal's "PQC core + classical head" option can then expand the output to the full feature vector.

### 8.7 Sizing for the hardware

| | This repo (MNIST) | Our Bot-IoT RBM (example) |
|---|---|---|
| Visible units | 784 | ~40 features + ~5 class labels ≈ 45 |
| Hidden units | 128 | ~16 |
| Couplers needed (`visible × hidden`) | 100,352 | ~720 |
| Fits a QPU after embedding? | No | Plausibly; check with `minorminer` before committing |

D-Wave's Advantage QPU has 5,000+ qubits, but a fully connected bipartite block needs chains, so the largest RBM that embeds cleanly is far smaller than the qubit count suggests. Test the embedding for your exact `(visible, hidden)` sizes early, since it may decide the RBM size for every condition.

---

## 9. Common confusions to avoid

**"Annealing finds the lowest energy" vs. "annealing samples the distribution."** These are different jobs. As an optimizer, an annealer returns the single best (lowest-energy) configuration. A GAN prior and the negative phase need **many different samples spread in proportion to their probability**. A prior that only returns ground states would collapse to a handful of attack patterns, the opposite of what we want for rare classes. D-Wave's **hybrid solvers** (the ones that handle 5,000+ variables) are built for optimization and aren't designed as Boltzmann samplers; the direct QPU with temperature estimation (§8.5) is the sampling route.

**Gaussian noise isn't an RBM condition.** "Bell-curve random noise" is the *standard GAN* baseline, in which no Boltzmann machine is involved at all. The RBM conditions differ from each other in **energy model** (classical RBM vs. QBM) and **sampler** (Gibbs/CD vs. annealing), as in the table in §3.4.

**0/1 vs. ±1 units.** Same model, different numbers (§3.3). Weights in one convention are not interchangeable with couplings in the other.

**Feature extraction vs. prior sampling.** mnist_example.py uses the RBM as an encoder of real data. We use it as a generator with no data input (§1).

**Reconstruction error vs. likelihood vs. sample quality.** Reconstruction error is a training progress signal. The RBM actually optimizes likelihood, which we can't compute exactly. What our pipeline cares about is sample quality, measured with our synthetic-data checks (§6.6).

**"Quantum-sampled RBM" vs. "QBM."** Sampling a classical RBM's energy on D-Wave (Γ = 0, Dixit et al. [4]) is a different experiment from training a model whose energy includes a transverse field (Γ ≠ 0). Results from one don't transfer to the other.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **Visible unit** | A unit representing observed data: a pixel here, a binarized traffic feature for us. |
| **Hidden unit** | A unit the model uses to explain the data. Learns to detect a pattern. |
| **Weight** | Strength of the connection between a visible and a hidden unit. Positive = they tend to be on together. |
| **Bias** | A unit's baseline tendency to be on. |
| **Energy** | A score for a configuration of all units. Lower energy = more probable. |
| **Partition function (Z)** | The normalizing constant summing over all configurations. Too large to compute. |
| **Sigmoid** | `1 / (1 + e^-x)`. Turns any number into a probability between 0 and 1. |
| **Gibbs sampling** | Alternately sampling all hidden units given the visible, then all visible given the hidden. |
| **Burn-in** | Gibbs steps discarded at the start of a chain so samples no longer depend on the starting point. |
| **Positive phase** | Statistics from real data ("reality"). |
| **Negative phase** | Statistics from the model's own samples ("dream"). The step quantum annealing replaces. |
| **Contrastive Divergence (CD-k)** | Training with a negative phase reached after only *k* Gibbs steps from real data. Fast but biased. |
| **Reconstruction error** | How far the model's dreamed input is from the real one. A progress indicator, not the objective. |
| **Momentum** | Carrying part of the previous update into the next one. |
| **Weight decay** | Shrinking weights slightly each step (L2 regularization). |
| **Prior** | The distribution the GAN generator's input `z` is drawn from: Gaussian in a standard GAN, Boltzmann in ours. |
| **Ising model** | Energy model over ±1 spins with fields and couplings. The form D-Wave hardware implements. |
| **QUBO** | Quadratic Unconstrained Binary Optimization: the same kind of model over 0/1 variables. |
| **Quantum annealing** | Slowly reducing a transverse field so qubits settle into low-energy configurations of an Ising model. |
| **Effective temperature (β_eff)** | The unknown scale at which a QPU's samples follow a Boltzmann distribution. Must be estimated to use the QPU as a sampler. |
| **Embedding / chain** | Representing one logical unit with several linked physical qubits so the problem fits the hardware graph. |
| **Transverse field (Γ)** | The `σˣ` term in the QBM Hamiltonian. Γ = 0 recovers a classical Boltzmann machine. |
| **Clamping** | Fixing some units (e.g. class labels) while sampling the rest. |
| **Macro-F1** | F1 averaged equally over classes, so rare classes count as much as common ones. |
| **PR-AUC** | Area under the precision–recall curve. More informative than ROC-AUC under heavy imbalance. |
| **Epoch / Batch** | One full pass over the training set / a group of examples processed in one update. |

### Further reading

- Geoffrey Hinton, *A Practical Guide to Training Restricted Boltzmann Machines* (2010). The reference for the tricks in rbm.py and for initialization.
- Geoffrey Hinton, *Training Products of Experts by Minimizing Contrastive Divergence* (2002). The paper that introduced CD.
- Steven Adachi & Maxwell Henderson, *Application of Quantum Annealing to Training of Deep Neural Networks* (2015), arXiv:1510.06356. Replacing the negative phase with D-Wave samples.
- Marcello Benedetti et al., *Estimation of Effective Temperatures in Quantum Annealers for Sampling Applications* (2016), arXiv:1510.07611. How to estimate β_eff.
- Mohammad Amin et al., *Quantum Boltzmann Machine* (2018), arXiv:1601.02036. Training with a transverse field.
- Dixit et al. [4] and Kim et al. [3] in our proposal's reference list.
