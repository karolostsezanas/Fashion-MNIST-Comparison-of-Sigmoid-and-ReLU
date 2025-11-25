# Two-Hidden-Layer MLPs on Fashion-MNIST
**Activation and Dropout under SGD**

Part of a Deep Learning assignment for an MSc in AI & Data Science. A two-hidden-layer multilayer perceptron (MLP) is trained and four variants are compared:

- **Sigmoid** (no dropout)
- **Sigmoid + Dropout**
- **ReLU** (no dropout)
- **ReLU + Dropout**

---

## Dataset
- **Fashion-MNIST**: 60,000 train / 10,000 test images  
- Grayscale **28×28**, flattened to **784** features  
- Pixel intensities scaled to **[0, 1]**

---

## Model & Training
- **Architecture**:  
  `Dense(256, act) → Dense(128, act) → Dense(10, softmax)`
- **Optimiser**: SGD (`learning_rate = 0.1`)
- **Loss**: Sparse Categorical Cross-Entropy
- **Metric**: Sparse Categorical Accuracy
- **Schedule**: 10 epochs, batch size 1000, validation split 0.1

> _Note:_ “act” is either `sigmoid` or `relu`. Dropout (if enabled) is applied after each hidden layer.

---

## Results (Test set)
| Rank | Model              | Test Acc | Test Loss |
|:---:|:-------------------|---------:|----------:|
| 1 | **ReLU_NoDropout**   | **0.8401** | **0.4508** |
| 2 | ReLU_Dropout         | 0.8366 | 0.4576 |
| 3 | Sigmoid_NoDropout    | 0.7212 | 0.8562 |
| 4 | Sigmoid_Dropout      | 0.6696 | 0.9632 |

**Summary:** ReLU improves optimisation and final accuracy over sigmoid. Under this short training schedule the models are not strongly overfitting, so introducing dropout slightly **reduces** accuracy for both activations.

---

## Repo Outline
- `notebooks/` – Colab/Jupyter notebook with full training and evaluation
- `models/` – Saved Keras models (`.keras`) for each variant 
- `README.md` – This file

---

