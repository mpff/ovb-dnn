# Controlling for Omitted Variable Bias in Deep Neural Networks

[![arXiv](https://img.shields.io/badge/arXiv-2608.25930-b31b1b.svg)](https://arxiv.org/abs/2608.25930)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

*Manuel Pfeuffer, Roshan P. Rane, Kerstin Ritter, Sonja Greven*

This repository contains the code for the paper *Controlling for Omitted Variable Bias in Deep Neural Networks*. It implements the proposed estimation procedure and the experiments of the paper.

We propose a control variable approach for deep neural networks (DNNs).
A DNN that predicts an outcome `Y` from an image `X` also learns from a covariate `Z`, such as age or sex, when `Z` is correlated with the outcome.
The network then attributes part of the effect of `Z` to `X`. This form of omitted variable bias is known as shortcut learning.
To control for this bias we propose a generalised additive model for the effects of the model input and the covariates.
The approach keeps the learned feature map `φ` of a pre-trained network fixed and refits the last layer as

```
g(E[Y | X, Z]) = η(X, Z) = β₀ + f_X(X) + f_Z(Z),    f_X(X) = φ(X)ᵀ β_X,    f_Z(Z) = Zᵀ β_Z.
```

Here, `g` is a link function and `f_X` is the direct effect of `X` with `Z` held fixed.

Flexible additive models can suffer from concurvity. For this reason, the estimation procedure refits only the last layer, with a ridge penalty `λ` and with cross-fitting. The pre-trained network and the refit use disjoint samples, and cross-fitting over `K` folds (Chernozhukov et al., 2018) recovers the full sample size. The refit uses backfitting, with iteratively reweighted least squares (IRLS) for a non-identity link. A validation split selects `λ`.

## Installation

```
git clone https://github.com/mpff/ovb-dnn.git
cd ovb-dnn
pip install -e .
```

To reproduce the experiments, use the conda environment in `environment.yml`. It also contains R for the figures.

## Example

The example below simulates data in which `Z` affects both `X` and `Y`. It trains a DNN without controls on each of two disjoint samples. It then refits the last layer of each network on the other sample, with `Z` as a control variable. The average of the two refits is the cross-fit estimate with `K = 2`.

```python
import torch
from torch.utils.data import ConcatDataset, DataLoader

from ovb_dnn.dataset import CovarDataset
from ovb_dnn.model import BaseNetwork
from ovb_dnn.refit_model import RefitCovarNetwork
from ovb_dnn.crossfit import CrossFitEnsemble
from ovb_dnn.trainer import covar_trainer


class Backbone(torch.nn.Module):          # phi: X -> R^q
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(in_features, 32), torch.nn.ReLU(),
                                       torch.nn.Linear(32, out_features), torch.nn.ReLU())
        self.out_features = out_features

    def forward(self, x):
        return self.net(x)


def sample(n):                            # Z drives both X and y: y = 2 X_1 + 3 Z + noise
    Z = torch.randn(n, 1)
    X = torch.randn(n, 10) + Z
    y = 2.0 * X[:, :1] + 3.0 * Z + 0.5 * torch.randn(n, 1)
    return X, Z, y


def loaders(n):
    ds_tr, ds_va = CovarDataset(*sample(n)), CovarDataset(*sample(n // 4))
    return DataLoader(ds_tr, batch_size=64, shuffle=True), DataLoader(ds_va, batch_size=256)


# 1. Train a DNN without controls on each of two disjoint samples A and B.
(train_A, val_A), (train_B, val_B) = loaders(2000), loaders(2000)
backbone = dict(backbone=Backbone, backbone_params=dict(in_features=10, out_features=8))
dnn_A = covar_trainer(BaseNetwork, backbone, train_A, val_A).center_effects(train_A)
dnn_B = covar_trainer(BaseNetwork, backbone, train_B, val_B).center_effects(train_B)

# 2. Refit each last layer with Z as a control variable on the other sample.
refit_A = RefitCovarNetwork(dnn_A, num_covariates=1).fit(train_B, val_B)
refit_B = RefitCovarNetwork(dnn_B, num_covariates=1).fit(train_A, val_A)

# 3. Cross-fit (K=2): average the two refits, centred on the full sample A ∪ B.
full = DataLoader(ConcatDataset([train_A.dataset, train_B.dataset]), batch_size=256)
model = CrossFitEnsemble([refit_A, refit_B]).recenter(full)

X, Z, _ = sample(500)
with torch.no_grad():
    f_x = model.predict_fx(X, Z)          # direct effect of X
    f_z = model.predict_fz(Z)             # effect of the control variable
    y_hat = model(X, Z)                   # g^{-1}(η)
```

For a single sample split, use one `RefitCovarNetwork` without the ensemble. For a binary outcome, set `link="logit"` in the model dictionary and pass `loss_fn=torch.nn.BCELoss()` to `covar_trainer`. With `orthogonalize=True`, a post-hoc regression also removes the part of `f_X` that is linear in `Z`. The result is the orthogonalized effect `f_X^re`.

## Backbones

The backbone is the network that computes the feature map `φ`. Any `torch.nn.Module` can be the backbone if it

- returns a `(batch, out_features)` tensor from `forward(x)`, and
- has an integer attribute `out_features`.

For an example, see the pre-trained 3D ResNet-50 for brain MRI in `experiments/ukbb/common/backbone.py`.

## Reproducing the paper

Each experiment script states at the top what it shows and which figures it produces. The R scripts in `figures/` draw the figures.

| Script | Shows | Paper figures |
|---|---|---|
| `experiments/simulation/study_a_linear_consistency.py` | The DNN with Controls recovers `f_X` as the confounding strength, backbone width, number of controls and sample size grow, while the DNN without controls absorbs `f_Z` | Fig 1, Fig 4, appendix |
| `experiments/simulation/study_b_misspecification.py` | A nonlinear `f_Z` refit as linear in `Z` | appendix |
| `experiments/simulation/study_c_concurvity_benchmark.py` | The DNN with Controls against neural additive models and post-hoc methods under concurvity | Fig 2, appendix |
| `experiments/simulation/study_d_noisy_z.py` | A control variable measured with noise | appendix |
| `experiments/ukbb/` | Brain MRI from the UK Biobank with synthetic confounding by age | Fig 5, appendix |

Run the simulations from the repository root, for example with `NSIM=5 python experiments/simulation/study_a_linear_consistency.py`. The script `experiments/simulation/aggregate.py` then reduces the predictions to CSV files in `experiments/simulation/output/`. The figures in the paper are drawn from these CSV files.

We cannot redistribute the UK Biobank data. Researchers can apply for access to the [UK Biobank](https://www.ukbiobank.ac.uk). The script `data/ukbb/make_dummy_data.py` writes random data files with the same layout. With these files, the UKBB pipeline runs end to end without access to the real data. The environment variable `UKBB_DATA_ROOT` points the pipeline to the real data. For details, see `experiments/ukbb/README.md`.

## Tests

```
pytest tests/
```

The tests in `tests/test_refit_model.py` compare the refit with known solutions. For an identity link, the ridge refit must match its closed-form solution, and with `λ = 0` it must match ordinary least squares. For a logit link, the IRLS refit must match the penalised maximum-likelihood estimate.

## Citation

```bibtex
@misc{pfeuffer2026controlling,
  title         = {Controlling for Omitted Variable Bias in Deep Neural Networks},
  author        = {Pfeuffer, Manuel and Rane, Roshan P. and Ritter, Kerstin and Greven, Sonja},
  year          = {2026},
  eprint        = {2608.25930},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ME},
  url           = {https://arxiv.org/abs/2608.25930}
}
```

## License

MIT, see [LICENSE](LICENSE).

The 3D ResNet in `experiments/ukbb/common/backbone.py` is adapted from [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch) (MIT License, Copyright (c) 2017 Kensho Hara; Hara et al., CVPR 2018). The UKBB experiment uses the ResNet-50 weights `r3d50_K_200ep.pth` from that repository, which are pre-trained on Kinetics-700.

## Acknowledgements

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation), project number 459422098. This research has been conducted using the UK Biobank resource under application number 33073.
