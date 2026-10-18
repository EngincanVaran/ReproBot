# extra-papers/ — CPU-sized replication targets beyond CIFAR-10

Papers whose headline experiment trains in seconds to about an hour on a CPU, chosen
so ReproBot can produce real fidelity numbers without a GPU and so the evaluation
covers **different model families and data types**, not just image CNNs. Kept
separate from `dataset/`, which is curated by someone else.

Every number below was read from the PDF itself (`pdfplumber`), not from a summary.
Every dataset URL was checked to respond (HTTP 200): the first batch on 2026-09-13, the
four added on 2026-09-20.

## Status

| Paper | Family | Status |
|---|---|---|
| Tang 2013 | MLP + L2-SVM loss, MNIST | full run done: 0.82% vs 0.87% (manual `C` fix) |
| Wijaya 2023 | MLP regression, Boston Housing | full run done: RMSE 4.48 vs 3.02 |
| SVM guide | RBF SVC | orchestrated: 96.625% vs 96.9% (implementation exact at paper's C, γ) |
| Fashion-MNIST | random forest | orchestrated: 0.8773 vs 0.873 (mean of 5) |
| Soft decision tree | gradient-trained tree | orchestrated: 95.11% vs 94.45% after fixing the paper's misprinted loss by hand |
| the other six | see tables | PDF added 2026-09-13; not yet OCR'd |
| the four below | classical ML, anomaly detection, two text | PDF added 2026-09-20; not yet OCR'd |

## The six added 2026-09-13

| File | Family / task | Suggested target claim | Data source | Needs |
|---|---|---|---|---|
| `2003 - A Practical Guide to Support Vector Classification.pdf` (Hsu, Chang, Lin; tech report, version updated 2025-09) | **SVM** (RBF SVC), tabular binary | svmguide1 test accuracy **96.875%** (scaled to [-1,1], C=2, γ=2). Also 66.925% unscaled default, 96.15% scaled default | LIBSVM datasets site, `binary/svmguide1` + `.t` (3,089 train / 4,000 test, 4 features) | **scikit-learn** support in `coder/`. `sklearn.svm.SVC` wraps LIBSVM, so this may reproduce *exactly* — the cleanest fidelity test available |
| `2017-08 - Fashion-MNIST - ...pdf` (Xiao, Rasul, Vollgraf; arXiv:1708.07747) | **Classical ML zoo** on image pixels: DecisionTree, ExtraTree, RandomForest, GradientBoosting, SVC, LinearSVC, kNN, LogisticRegression, MLPClassifier, GaussianNB, Perceptron, SGD | Table 3, mean of 5 shuffled runs. E.g. DecisionTree (entropy, depth 10) **0.798**; RandomForest (100, entropy, depth 100) **0.873**; SVC (C=10, rbf) **0.897**; MLP (relu, [100]) **0.871**; kNN (distance, k=5, p=1) **0.854** | `torchvision.datasets.FashionMNIST` (also OpenML) | **scikit-learn** support. 2017 sklearn defaults differ from our 1.5.2 (e.g. `loss=deviance` removed, SVC `gamma` default changed) — a library-version-defaults test like the Keras one. Table has ~130 rows: watch the Reader's claims output size. SVC on 60k rows × 5 runs is hours; trees/forests are minutes |
| `2016-09 - Semi-Supervised Classification with Graph Convolutional Networks.pdf` (Kipf, Welling; ICLR 2017, arXiv:1609.02907) | **Graph neural network** | Cora accuracy **81.5%** (mean of 100 random inits). Every hyperparameter stated: Adam lr 0.01, ≤200 epochs, early stop window 10, 16 hidden, dropout 0.5, L2 5e-4, Glorot init; split 20 labels/class, 500 val, 1,000 test | Planetoid raw files, `github.com/kimiyoung/planetoid/raw/master/data/ind.cora.*` (Python pickles; needs scipy, already in the image) | Works with the pipeline as is. Seconds per run, so the 100-run mean is cheap — a good test for a variance-aware Critic |
| `2014-08 - Convolutional Neural Networks for Sentence Classification.pdf` (Kim; EMNLP 2014, arXiv:1408.5882) | **Text CNN** (NLP) | CNN-rand (no pretrained vectors) TREC accuracy **91.2%** (standard 500-question test split). MR 76.1% needs 10-fold CV | TREC: `cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label`, `TREC_10.label`; MR: Cornell `rt-polaritydata.tar.gz` | Works as is. Filters 3/4/5 × 100, dropout 0.5, max-norm 3, batch 50, Adadelta stated; epochs not (early stopping on a 10% dev split). Minutes on CPU |
| `2018-03 - An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.pdf` (Bai, Kolter, Koltun; arXiv:1803.01271) | **Sequence model** (TCN vs LSTM/GRU), synthetic regression | Adding problem T=600, TCN test MSE **5.8e-5** (k=8, n=8, hidden 24, no dropout, no clip, Adam lr 0.002). Copy memory T=1000 loss 3.5e-5 | **Synthetic** — generated in code, no download | Small prompt addition: data generated from a stated procedure. Training-set size not stated in the paper. Estimated 30–90 min on CPU |
| `2017-11 - Distilling a Neural Network Into a Soft Decision Tree.pdf` (Frosst, Hinton; arXiv:1711.09784) | **Soft decision tree** (tree trained by gradient descent) | MNIST, depth-8 soft tree on true labels: "at most" **94.45%**; distilled from a CNN: 96.76% | `torchvision.datasets.MNIST` | **Stretch target.** Learning rate, penalty strength λ, inverse temperature β, epochs and batch size are all unstated — a stress test of the unstated-details discipline, not a clean fidelity test |

## Gradient boosting libraries, added 2026-09-13

| File | Family / task | Suggested target claim | Data source | Needs |
|---|---|---|---|---|
| `2019-11 - A Comparative Analysis of XGBoost.pdf` (Bentéjac, Csörgő, Martínez-Muñoz; arXiv:1911.01914) | **XGBoost vs random forest vs gradient boosting**, 28 small tabular datasets | Table 3, stratified 10-fold CV, 200 trees, default and grid-searched parameters (Table 2 lists both). E.g. Spambase: default XGBoost **95.17% ± 1.29**, default RF 95.46%, tuned GB 96.11% | UCI datasets, most on OpenML (Spambase = OpenML 44) | **xgboost** in the Runner image; scikit-learn support in `coder/`. The paper's "default" XGBoost is 2019's (`max_depth=3`, `learning_rate=0.1`); xgboost 2.x defaults differ — another library-version-defaults test. Seconds per dataset |
| `2017-06 - CatBoost - Unbiased Boosting with Categorical Features.pdf` (Prokhorenkova et al.; NeurIPS 2018, arXiv:1706.09516) | **CatBoost vs LightGBM vs XGBoost**, tabular classification with categorical features | Appendix Table 8, log loss / zero-one loss on Adult: CatBoost **0.2695 / 0.1267**, LightGBM **0.2760 / 0.1291**, XGBoost **0.2754 / 0.1280** | Adult = OpenML 1590; Amazon = OpenML 4135 | **xgboost, lightgbm, catboost** (and `hyperopt`) in the image. Protocol is demanding: 80/20 split, ordered target statistics for categoricals, 50 TPE tuning steps on 5-fold CV. **Advanced target** — the best available LightGBM claim on CPU-sized data |
| `2016-03 - XGBoost - A Scalable Tree Boosting System.pdf` (Chen, Guestrin; KDD 2016, arXiv:1603.02754) | **XGBoost** (exact greedy), large tabular classification | Table 3, Higgs-1M: XGBoost test AUC **0.8304** (500 trees, max depth 8, shrinkage 0.1, no column subsampling); scikit-learn GBM 0.8302 | UCI HIGGS, `archive.ics.uci.edu/static/public/280/higgs.zip` (**~2.6 GB**, responds) | **xgboost** in the image. The 1M subset and the test split are not specified. The XGBoost claim is ~minutes; the scikit-learn claim is hours (28.5 s/tree on their 16 cores) — skip it |

## Four added 2026-09-20 — classical ML, anomaly detection, text

Chosen to widen the evaluation in three directions at once: **cheap** (all four train in
seconds to minutes on a CPU), **classical** (three need no neural network at all), and
**two new task types** — anomaly detection scored by AUC, and text classification. Three
of the four also state a claim **per dataset**, so one paper gives many independent
fidelity tests rather than one.

| File | Family / task | Suggested target claim | Data source | Needs |
|---|---|---|---|---|
| `2001-10 - Random Forests.pdf` (Breiman; *Machine Learning* 45(1), 2001) | **Random forest**, 13 small tabular datasets | Table 2, Forest-RI "selection" test error: glass **20.6%**, breast cancer **2.9%**, diabetes **24.2%**, sonar **15.9%**, vowel **3.4%**, ionosphere **7.1%**, vehicle **25.8%**, German credit **24.4%**, image **2.1%**, ecoli **12.8%**, votes **4.1%**, liver **25.1%**, letters **3.5%** | OpenML, every one pinned: glass 41, diabetes 37, sonar 40, ionosphere 59, vehicle 54, credit-g 31, segment 36, ecoli 39, vote 56, liver-disorders 8, letter 6 | Works with the pipeline as is. **The protocol is fully stated and unusually favourable**: hold out a random 10%, grow 100 trees twice (F=1 and F=int(log₂M+1)), keep whichever has the lower out-of-bag error, repeat 100 times and average. That gives the Critic a real measured spread instead of a binomial proxy, and 13 claims from one paper. Seconds per dataset |
| `2008-12 - Isolation Forest.pdf` (Liu, Ting, Zhou; ICDM 2008) | **Anomaly detection** (unsupervised scoring, evaluated by AUC) | Table 2 AUC at the stated defaults (t=100 trees, ψ=256 sub-sample): Shuttle **1.00**, Breastw **0.99**, ForestCover **0.88**, Smtp **0.88**, Mammography **0.86**, Annthyroid **0.82**, Satellite **0.71**, Pima **0.67**, Http **1.00** | OpenML: shuttle 40685, mammography 310, satimage 182, diabetes (Pima) 37, breast-w 15. Http/Smtp/ForestCover come from `sklearn.datasets.fetch_kddcup99` and `fetch_covtype` | `sklearn.ensemble.IsolationForest(n_estimators=100, max_samples=256)` is this algorithm by the same authors' design. **A new task shape**: no train/test label split, the metric is ranking quality (AUC, higher is better) over a contaminated set. Worth checking the Critic handles a metric with no train split. Seconds |
| `2016-07 - Bag of Tricks for Efficient Text Classification.pdf` (Joulin, Grave, Bojanowski, Mikolov; arXiv:1607.01759) | **Text classification**, linear bag-of-n-grams | Table 1 test accuracy, h=10 with bigrams: AG News **92.5%**, DBpedia **98.6%**, Yelp Polarity **95.7%**, Yahoo Answers **72.3%**. Table 2 also claims **1 s** per epoch on AG with one CPU thread | AG News CSV mirror (`raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/`, 120k train / 7.6k test), DBpedia from the same Zhang et al. 2015 collection | The model *is* an `EmbeddingBag` + linear layer + softmax, so PyTorch reproduces it faithfully; 5 epochs, 10 hidden units, bigrams hashed into buckets, linearly decaying learning rate (its value "selected on a validation set", i.e. unstated). **Two firsts for the pipeline**: a text corpus fetched from a plain URL, and a paper that claims a *speed* as well as an accuracy |
| `2002-07 - Thumbs up - Sentiment Classification using Machine Learning Techniques.pdf` (Pang, Lee, Vaithyanathan; EMNLP 2002) | **Text classification**, the classic NB/MaxEnt/SVM comparison | Figure 3, three-fold CV accuracy on 1,400 movie reviews, unigrams with **presence** (not frequency): SVM **82.9%**, Naive Bayes **81.0%**, MaxEnt **80.4%**. Frequency-weighted unigrams drop SVM to 72.8% — a sharp, checkable contrast | Cornell polarity dataset, `cs.cornell.edu/people/pabo/movie-review-data/` | Pure scikit-learn (`LinearSVC`, `MultinomialNB`), seconds to run. The 16,165-feature vocabulary and the balanced 700/700 sample are stated; the fold assignment is not. **The best small test of "does presence beat frequency"** — a claim about a *design choice*, not just a number |

### Why these four, specifically

- **They stress the Critic differently.** Breiman's 100-repeat protocol produces a genuine
  run distribution, so the tolerance comes from measured spread rather than the binomial
  stand-in. Isolation Forest has no train/test split at all. fastText claims a runtime.
- **They are honest about difficulty.** Three need no GPU, no framework translation and no
  unstated architecture: if ReproBot cannot reproduce Breiman's 2.9% on breast cancer,
  the problem is ReproBot, not the compute budget.
- **They multiply the evaluation cheaply.** Breiman alone offers 13 independent claims and
  Isolation Forest 9, against one claim per paper today.

## Considered and left out

- **KMNIST** (Clanuwat et al. 2018, arXiv:1812.01718): baseline table exists, but the
  paper defers all training setup to its GitHub repository.
- **LeNet-5** (LeCun et al. 1998): the canonical small CNN, but 46 pages — expensive to
  OCR one page per call for one claim.
- **Dropout / maxout MNIST results**: need hundreds to thousands of epochs — too slow on CPU.
- **LightGBM** (Ke et al., NeurIPS 2017): every dataset in it has 2M+ rows (Allstate,
  Flight Delay, LETOR, KDD10/12). LightGBM is covered instead through the CatBoost
  paper's baselines.
- **Tabular benchmarks with large HPO budgets** (Grinsztajn et al. 2022; Shwartz-Ziv &
  Armon 2021; McElfresh et al. 2023): claims come out of hundreds of tuning runs per
  dataset, mostly reported as plots or ranks.
- **Unsupervised papers** (k-means, autoencoders, VAEs): ReproBot's stages assume a
  supervised claim on a held-out split; out of scope until that changes. Isolation Forest
  above is the deliberate first exception: unsupervised training, but a scored claim.
- **SMOTE** (Chawla et al. 2002, JAIR): CPU-cheap and the datasets are on OpenML, but the
  results are ROC curves and convex hulls rather than a table of numbers to diff against.
- **Learning Word Vectors for Sentiment Analysis** (Maas et al. 2011): IMDB 88.89% is a
  clean claim, but their model trains sentiment-aware word vectors first — much heavier
  than the other three text targets. Revisit after fastText.
- **Support-Vector Networks** (Cortes & Vapnik 1995): the USPS error rates would be an
  excellent target, but no open PDF responded.
- **Extremely Randomized Trees** (Geurts et al. 2006): open PDF, but its tables report
  complexity and tuned protocols per dataset rather than a single stated setup.
