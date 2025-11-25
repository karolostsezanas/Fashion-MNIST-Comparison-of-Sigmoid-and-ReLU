Two-Hidden-Layer MLPs on Fashion-MNIST

Activation and Dropout under SGD

Part of a Deep Learning assignment for an MSc in AI and Data Science.
Train a two-hidden-layer MLP and compare four variants:
Sigmoid w/wo Dropout and ReLU w/wo Dropout.

Dataset: Fashion-MNIST (60k train, 10k test), grayscale 28×28, flattened to 784 features and scaled to [0, 1].

Optimiser: SGD (lr=0.1). Loss: Sparse Categorical Cross-Entropy. Metric: Sparse Categorical Accuracy.

Architecture: Dense(256, act) → Dense(128, act) → Dense(10, softmax).

Results (10 epochs, batch size 1000)
Rank	Model	Test Acc	Test Loss
1	ReLU_NoDropout	0.8401	0.4508
2	ReLU_Dropout	0.8366	0.4576
3	Sigmoid_NoDropout	0.7212	0.8562
4	Sigmoid_Dropout	0.6696	0.9632

ReLU improves optimisation and final accuracy. Under this short training schedule the networks are not strongly overfitting, so dropout slightly hurts both activations.
