# extra-papers/ — CPU-sized replication targets beyond CIFAR-10

Papers whose headline experiment trains in seconds to about an hour on a CPU, chosen
so ReproBot can produce real fidelity numbers without a GPU and so the evaluation
covers **different model families and data types**, not just image CNNs. Kept
separate from `dataset/`, which is curated by someone else.

Every number below was read from the PDF itself (`pdftotext`), not from a summary.
Every dataset URL was checked to respond (HTTP 200) on 2026-09-13.

## Status

| Paper | Family | Status |
|---|---|---|
| Tang 2013 | MLP + L2-SVM loss, MNIST | full run done: 0.82% vs 0.87% (manual `C` fix) |
| Wijaya 2023 | MLP regression, Boston Housing | full run done: RMSE 4.48 vs 3.02 |
| the six below | see table | PDF added 2026-09-13; not yet OCR'd |

## The six added 2026-09-13

| File | Family / task | Suggested target claim | Data source | Needs |
|---|---|---|---|---|
| `2003 - A Practical Guide to Support Vector Classification.pdf` (Hsu, Chang, Lin; tech report, version updated 2025-09) | **SVM** (RBF SVC), tabular binary | svmguide1 test accuracy **96.875%** (scaled to [-1,1], C=2, γ=2). Also 66.925% unscaled default, 96.15% scaled default | LIBSVM datasets site, `binary/svmguide1` + `.t` (3,089 train / 4,000 test, 4 features) | **scikit-learn** support in `coder/`. `sklearn.svm.SVC` wraps LIBSVM, so this may reproduce *exactly* — the cleanest fidelity test available |
| `2017-08 - Fashion-MNIST - ...pdf` (Xiao, Rasul, Vollgraf; arXiv:1708.07747) | **Classical ML zoo** on image pixels: DecisionTree, ExtraTree, RandomForest, GradientBoosting, SVC, LinearSVC, kNN, LogisticRegression, MLPClassifier, GaussianNB, Perceptron, SGD | Table 3, mean of 5 shuffled runs. E.g. DecisionTree (entropy, depth 10) **0.798**; RandomForest (100, entropy, depth 100) **0.873**; SVC (C=10, rbf) **0.897**; MLP (relu, [100]) **0.871**; kNN (distance, k=5, p=1) **0.854** | `torchvision.datasets.FashionMNIST` (also OpenML) | **scikit-learn** support. 2017 sklearn defaults differ from our 1.5.2 (e.g. `loss=deviance` removed, SVC `gamma` default changed) — a library-version-defaults test like the Keras one. Table has ~130 rows: watch the Reader's claims output size. SVC on 60k rows × 5 runs is hours; trees/forests are minutes |
| `2016-09 - Semi-Supervised Classification with Graph Convolutional Networks.pdf` (Kipf, Welling; ICLR 2017, arXiv:1609.02907) | **Graph neural network** | Cora accuracy **81.5%** (mean of 100 random inits). Every hyperparameter stated: Adam lr 0.01, ≤200 epochs, early stop window 10, 16 hidden, dropout 0.5, L2 5e-4, Glorot init; split 20 labels/class, 500 val, 1,000 test | Planetoid raw files, `github.com/kimiyoung/planetoid/raw/master/data/ind.cora.*` (Python pickles; needs scipy, already in the image) | Works with the pipeline as is. Seconds per run, so the 100-run mean is cheap — a good test for a variance-aware Critic |
| `2014-08 - Convolutional Neural Networks for Sentence Classification.pdf` (Kim; EMNLP 2014, arXiv:1408.5882) | **Text CNN** (NLP) | CNN-rand (no pretrained vectors) TREC accuracy **91.2%** (standard 500-question test split). MR 76.1% needs 10-fold CV | TREC: `cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label`, `TREC_10.label`; MR: Cornell `rt-polaritydata.tar.gz` | Works as is. Filters 3/4/5 × 100, dropout 0.5, max-norm 3, batch 50, Adadelta stated; epochs not (early stopping on a 10% dev split). Minutes on CPU |
| `2018-03 - An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.pdf` (Bai, Kolter, Koltun; arXiv:1803.01271) | **Sequence model** (TCN vs LSTM/GRU), synthetic regression | Adding problem T=600, TCN test MSE **5.8e-5** (k=8, n=8, hidden 24, no dropout, no clip, Adam lr 0.002). Copy memory T=1000 loss 3.5e-5 | **Synthetic** — generated in code, no download | Small prompt addition: data generated from a stated procedure. Training-set size not stated in the paper. Estimated 30–90 min on CPU |
| `2017-11 - Distilling a Neural Network Into a Soft Decision Tree.pdf` (Frosst, Hinton; arXiv:1711.09784) | **Soft decision tree** (tree trained by gradient descent) | MNIST, depth-8 soft tree on true labels: "at most" **94.45%**; distilled from a CNN: 96.76% | `torchvision.datasets.MNIST` | **Stretch target.** Learning rate, penalty strength λ, inverse temperature β, epochs and batch size are all unstated — a stress test of the unstated-details discipline, not a clean fidelity test |

## Considered and left out

- **KMNIST** (Clanuwat et al. 2018, arXiv:1812.01718): baseline table exists, but the
  paper defers all training setup to its GitHub repository.
- **LeNet-5** (LeCun et al. 1998): the canonical small CNN, but 46 pages — expensive to
  OCR one page per call for one claim.
- **Dropout / maxout MNIST results**: need hundreds to thousands of epochs — too slow on CPU.
- **Unsupervised papers** (k-means, autoencoders, VAEs): ReproBot's stages assume a
  supervised claim on a held-out split; out of scope until that changes.
