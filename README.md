# Project Azua

## 0. Overview

Many modern AI algorithms are known to be data-hungry, whereas human decision-making is much 
more efficient. The human can reason under uncertainty, actively acquire valuable information from the world to reduce 
uncertainty, and make personalized decisions given incomplete information. How can we replicate those abilities in machine
intelligence?

In project Azua, we build AI algorithms to aid efficient 
decision-making with minimum data requirements. To achieve optimal trade-offs and enable the human in the loop, we 
combine state-of-the-art methods in deep learning, probabilistic inference, and causality. We provide easy-to-use deep 
learning tools that can perform efficient multiple imputation under partially observed, mixed type data, discover the 
underlying causal relationship behind the data, and suggest the next best step for decision making. Our technology has 
enabled personalized decision-making in real-world systems, wrapping multiple advanced research in simple APIs, suitable 
for research development in the research communities, and commercial usages by data scientists and developers.

### References

If you have used the models in our code base, please consider to cite the corresponding paper:

1, **(PVAE and information acquisition)** Chao Ma, Sebastian Tschiatschek, Konstantina Palla, Jose Miguel Hernandez-Lobato, Sebastian Nowozin, and Cheng Zhang. "EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE." In International Conference on Machine Learning, pp. 4234-4243. PMLR, 2019.

2, **(VAEM)** Chao Ma, Sebastian Tschiatschek, Richard Turner, José Miguel Hernández-Lobato, and Cheng Zhang. "VAEM: a Deep Generative Model for Heterogeneous Mixed Type Data." Advances in Neural Information Processing Systems 33 (2020).

3, **(VICause)** Pablo Morales-Alvarez, Angus Lamb, Simon Woodhead, Simon Peyton Jones, Miltos Allamanis, and Cheng Zhang and Cheng Zhang, "VICAUSE: Simultaneous missing value imputation and causal discovery",
ICML 2021 workshop on the Neglected Assumptions in Causal Inference

4, **(Eedi dataset)** Zichao Wang, Angus Lamb, Evgeny Saveliev, Pashmina Cameron, Yordan Zaykov, Jose Miguel Hernandez-Lobato, Richard E. Turner et al. "Results and Insights from Diagnostic Questions: The NeurIPS 2020 Education Challenge." arXiv preprint arXiv:2104.04034 (2021).

### Resources

For quick introduction to our work, checkout [our NeurIPS 2020 tutorial, from 2:17:11](https://slideslive.com/38943571/advances-in-approximate-inference).
For a more in-depth technical introduction, checkout [our ICML 2020 tutorial](https://slideslive.com/38931299/efficient-missingvalue-acquisition-with-variational-autoencoders?ref=account-folder-55866-folders)

## 1. Core functionalities

Azua has there core functionalities: missing data imputation, active information acquisition, and causal discovery. 

**TODO**: Add image

### 1.1. Missing data imputation (MDI)

In many real-life scenarios, we will need to make decisions under incomplete information. Therefore, it is crucial to 
make accurate "guesses" regarding the missing information. To this end, Azua provides state-of-the-art missing
data imputation methods based on deep learning algorithms. These methods are able to learn from complete data, and then 
perform missing value imputation. Instead of producing only one single values for missing entries
as in common softwares, most of our methods are able to return multiple imputation values, which provide valuable 
information of imputation uncertainty.  

**TODO**: should we add description of different split mode of data imputation here? (element vs row)

### 1.2. Personalized active information acquisition (PIA)

Azua can not only be used as a powerful data imputation tool, but also as an engine
to actively acquire valuable information. Given an incomplete data instance, Azua is able to suggest which unobserved 
variable is the most informative one (subject to the specific data instance and the task) to collect, using 
information-theoretic approaches. This allows the users to focus on collecting only the most important information, 
and thus make decisions with minimum amount of data. 

Our active information acquisition functionality has two modes: i) if there is a specific variable (called target 
variable) that the user wants to predict, then Azua will suggest the next variable to collect, that is most valuable to 
predicting that particular target variable. ii) otherwise, Azua will make decisions using built-in criterion.   

### 1.3 Causal discovery under missing data (CD)

The underlying causal relationships behind data crucial for real-world decision making. However, discovering causal
structures under incomplete information is difficult. Azua provide state-of-the-art solution based on graph neural nets,
 which is able to tackle missing value imputation and causal discovery problems at the same time.

## 2. Getting started

**TODO (MK)**: Make it shorter, remove unnecessary details

### Set-up Python environement

A [conda environment](environment.yml) is used to manage system requirements. To install conda, check the
installation instructions [here](https://docs.conda.io/en/latest/miniconda.html).
To create the azua environment, after initializing submodules as above, run

```bash
conda env create -f environment.yml
```

And to activate the environment run

```bash
conda activate azua
```

### Run experiment 

[`run_experiment.py`](run_experiment.py) script runs any combination of model training, imputation and active learning.

**TODO (MK)**: Explain how to get the dataset

The simplest way to run this script is using an existing dataset:
e.g.

```bash
python run_experiment.py boston -i -a eddi rand
```
In this example, we train a PVAE model on the Boston Housing dataset, evaluate the imputation performance on the test
set (the -i option) and compare the sequential feature acquisition performance between the EDDI policy and random policy.

To run on sparse-format data, where the dataset rows are of the form `(row_id, column_id, value)`, as is common in 
recommender system data, specify the `dataset_format` as `"sparse_csv"` in
`configs/[dataset_name]/dataset_config.json`.

To overwrite config values, specify model config (`-m`), train config (`-t`), imputation config (`-ic`), or objective 
config (`-oc`). These should be JSON files containing values to override from the default values found first in 
`configs/defaults` and then in `configs/[dataset_name]`.
e.g.

```bash
python run_experiment.py boston -m configs/model_config_sweep.json
```

Other useful options include:

* `-o`: change the output directory. This defaults to `./runs`
* `-n`: give a name to the ouput folder. This defaults to `./runs/$current_datetime`, but giving a
  name overrides to `./runs/$name_$current_datetime`.
* `-d`: the data directory. Defaults to `./data`.

## 3. Model overview

Below we summarize the list of models currently available in Azua, their descriptions, functionalities (MDI = missing 
data imputation, PIA = personalized information acquisition, CD = Causal discovery), and an example code that shows how 
to run the model (which will also reproduce one experiment from the paper). 

Model | Description | Functionalities | Example usage 
--- | --- | --- | --- 
[Partial VAE (PVAE)](models/partial_vae.py) | An extension of VAEs for partially observed data. See [our paper](https://arxiv.org/abs/1809.11142). |  MDI, PIA | `python run_experiment.py boston -mt pvae -a eddi rand`
[VAE Mixed (VAEM)](models/vae_mixed.py) | An extension of PVAE for heterogeneous mixed type data. See [our paper](https://arxiv.org/abs/1809.11142). | MDI, PIA | `python run_experiment.py bank -mt vaem_predictive -a sing` 
[MNAR Partial VAE (MNAR-PVAE)](models/mnar_pvae.py) | An extension of VAE that handles missing-not-at-random (MNAR) data. More details in the future. | MDI, PIA | `python run_experiment.py yahoo -mt mnar_pvae -i` 
[Bayesian Partial VAE (PVAE)](models/bayesian_pvae.py) | **TODO**: short info + link to paper | MDI, PIA | **TODO**: example usage to reproduce one experiment from the paper 
[Transformer PVAE](models/transformer_pvae.py) | A PVAE in which the encoder and decoder contain transformers. See [our paper](**TODO**)| MDI, PIA | `python run_experiment.py boston -mt transformer_pvae -a eddi rand`
[Transformer encoder PVAE](models/transformer_encoder_pvae.py) | A PVAE in which the encoder contains a transformer. See [our paper](**TODO**) | MDI, PIA | `python run_experiment.py boston -mt transformer_encoder_pvae -a eddi rand`
[Transformer imputer/Rupert](models/transformer_imputer.py) | A simple transformer model. See [our paper](**TODO**) | MDI, PIA | `python run_experiment.py boston -mt transformer_imputer -a variance rand`
[VICause](models/vicause.py) | Causal discovery from data with missing features and imputation. [link to paper](https://www.microsoft.com/en-us/research/publication/vicause-simultaneous-missing-value-imputation-and-causal-discovery/). | MDI, PIA, CD | **TODO**: example usage to reproduce one experiment from the paper 
[CoRGi](models/corgi.py) |  GNN-based imputation with emphasis on item-related data based on [Kim et al.](https://toappear) | MDI | **TODO**: example usage to reproduce one experiment from the paper 
[Graph Convolutional Network (GCN)](models/graph_convolutional_network.py) |  GNN-based imputation based on [Kipf et al.](https://arxiv.org/abs/1609.02907) | MDI | **TODO**: example usage to reproduce one experiment from the paper 
[GRAPE](models/grape.py) |  GNN-based imputation based on [You et al.](https://snap.stanford.edu/grape/) | MDI | **TODO**: example usage to reproduce one experiment from the paper 
[Graph Convolutional Matrix Completion (GC-MC)](models/gcmc.py) |   GNN-based imputation based on [van den Berg et al.](https://arxiv.org/abs/1706.02263) | MDI | **TODO**: example usage to reproduce one experiment from the paper 
[GraphSAGE](models/graphsage.py) |  GNN-based imputation based on [Hamilton et al.](https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html) | MDI | **TODO**: example usage to reproduce one experiment from the paper 
[Graph Attention Network (GAT)](models/graph_attention_network.py) | Attention-based GNN imputation based on [Veličković et al.](https://arxiv.org/abs/1710.10903) | MDI | **TODO**: example usage to reproduce one experiment from the paper 
[Mean imputing](baselines/mean_imputing.py) | **TODO**: short info + link to paper | **TODO**: functionalities | **TODO**: example usage to reproduce one experiment from the paper 
[Zero imputing](baselines/zero_imputing.py) | **TODO**: short info + link to paper | **TODO**: functionalities | **TODO**: example usage to reproduce one experiment from the paper 
[Min imputing](baselines/min_imputing.py) | **TODO**: short info + link to paper | **TODO**: functionalities | **TODO**: example usage to reproduce one experiment from the paper 
[Majority vote](baselines/majority_vote.py) | **TODO**: short info + link to paper | **TODO**: functionalities | **TODO**: example usage to reproduce one experiment from the paper 
[MICE](baselines/mice.py) | **TODO**: short info + link to paper | **TODO**: functionalities | **TODO**: example usage to reproduce one experiment from the paper 
[MissForest](baselines/missforest.py) | **TODO**: short info + link to paper | **TODO**: functionalities | **TODO**: example usage to reproduce one experiment from the paper 

**TODO**: Should we list here all Transformer models too?

**TODO**: Should we list here DMF? Should we list here BNN?

## Objectives

Objective | Description
--- | ---
[EDDI](objectives/eddi.py) | It uses information gain given observed values to predict the next best feature to query.
[SING](objectives/sing.py) | It uses a fixed information gain ordering based on no questions being asked.
[Random](objectives/rand.py) | It randomly selects the next best feature to query.
[Variance](objectives/variance.py) | It queries the feature that is expected to reduce predictive variance in the target variable the most.
**TODO**: Add other objectives

## 4. Reference results

### Missing data imputation: Test Data Normaliized RMSE

Dataset | Partial VAE | **TODO**
--- | --- | ---
Bank | 0.51 | **TODO**
**TODO** | **TODO** | **TODO**

**TODO**: Add more models/Datasets

### Next best question: area under information curve (AUIC)

**TODO**: More explanation what AUIC is

Dataset | Partial VAE | **TODO**
--- | --- | ---
Bank | 6.6 | **TODO**
**TODO** | **TODO** | **TODO**

**TODO**: Add more models/Datasets

## 5. Model details

### 5.1 Partial VAE

**Model Description**

Partial VAE (PVAE) is an unsupervised deep generative model, that is specifically designed to handle missing data. We mainly
use this model to learn the underlying structure (latent representation) of the partially observed data, and perform missing data
imputation. Just like any vanilla VAEs, PVAE is comprised of an encoder and a decoder. The PVAE encoder is parameterized 
by the so-called set-encoder (point-net, see [our paper](https://arxiv.org/abs/1809.11142) for details), which is able 
to extract the latent representation from partially observed data. Then, the PVAE decoder can take as input the extracted 
latent representation, and generate values for both missing entries (imputation), and observed entries (reconstruction).  

**The partial encoder**

One of the major differences between PVAE and VAE is, the PVAE encoder can handle missing data in a principled way. The 
PVAE encoder is parameterized by the so-called set-encoder, which will process partially observed data in three steps: 
1, feature embedding; 2, permutation-invariant aggregation; and 3, encoding into statistics of latent representation. 
These are implemented in [`feature_embedder.py`](models/feature_embedder.py), ['point_net.py'](models/point_net.py), and 
[`encoder.py`](models/encoder.py), respectively. see [our paper, Section 3.2](https://arxiv.org/abs/1809.11142) for technical details. 


**Model configs**

* `"embedding_dim"`: dimensionality of embedding for each input to PVAE encoder. 
    See [our paper](https://arxiv.org/abs/1809.11142) for details.
* `"set_embedding_dim"`: dimensionality of output set embedding in PVAE encoder.
    See [our paper](https://arxiv.org/abs/1809.11142) for details.
* `"set_embedding_multiply_weights"`: Whether or not to take the product of x with embedding weights when feeding through. 
Default: `true`.
* `"latent_dim"`: dimensionality of the PVAE latent representation
* `"encoder_layers"`: structure of encoder network (excluding input and output layers)
* `"decoder_layers"`: structure of decoder network (excluding input and output layers)
* `"non_linearity"`: Choice of non-linear activation functions for hidden layers of PVAE decoder. Possible choice: 
`"ReLU"`, `"Sigmoid"`, and `"Tanh"`. Default is `"ReLU"`.
* `"activation_for_continuous"`: Choice of non-linear activation functions for the output layer of PVAE decoder. 
Possible choice: `"Identity"`, ```"ReLU"`, `"Sigmoid"`, and `"Tanh"`. Default is `"Sigmoid"`.
* `"init_method"`: Initialization method for PVAE weights. Possible choice: `"default"` (Pytorch default), 
`"xavier_uniform"`, `"xavier_normal"`, `"uniform"`, and `"normal"`. Default is `"default"`.
* `"encoding_function"`: The permutation invariant operator of PVAE encoder. Default is `"sum"`.
* `"decoder_variances"`: Variance of the observation noise added to the PVAE decoder output (for continuous variables 
only).
* `"random_seed"`: Random seed used for initializing the model. Default: `[0]`.
* `"categorical_likelihood_coefficient"`: The coefficient for the likelihood terms of categorical variables. 
Default: `1.0`.
* `"kl_coefficient"`: The Beta coefficient for the KL term. Default: `1.0`.
* `"variance_autotune"`: Automatically learn the variance of the observation noise or not. Default: `false`.
* `"use_importance_sampling"`: Use importance sampling or not, when calculating the PVAE ELBO. When turned on, the PVAE 
will turn into importance weighted version of PVAE. See [IWAE](https://arxiv.org/abs/1509.00519) for more details. 
Default: `false`,
* `"squash_input"`: When preprocessing the data, squash the data to be between 0 and 1 or not. Default: `true`. Note 
that when `false`, you should change the config of `"activation_for_continuous"` accordingly (from `"Sigmoid"` to 
`"Identity"`).

### 5.2 VAEM 

**Model Description**

Real-world datasets often contain variables of different types (categorical, ordinal, continuous, etc.), and different
 marginal distributions. Although PVAE is able to cope with missing data, it does not handle heterogeneous mixed-type data 
 very well. Azua provide a new model called VAEM to handle such scenarios. 
 
**The marginal VAEs and the dependency network**
 
In short, VAEM is an extension to VAE that can handle such heterogeneous data. It is a deep generative model that is 
trained in a two stage manner. 

* In the first stage, we model the marginal distributions of each single variable separately. 
This is done by fitting  a different vanilla VAE independently to each data dimension. This is implemented in 
[`marginal_vaes.py`](models/marginal_vaes.py). Those one-dimensional VAEs will capture the marginal properties of each 
variable and provide a latent representation that is more homogeneous across dimensions. 

* In the second stage, we capture  the dependencies among each variables. To this end, another Partial VAE, called the 
dependency network, is build on top of the latent representations provided by the first-stage VAEs. This is implemented 
in [`dependency_network_creator`](models/dependency_network_creator.py)

To summarize, we can think of the first stage of VAEM as a data pre-processing step, that transforms heterogeneous 
mixed-type data into a homogeneous version of the data. Then, we can perform missing data imputation and personalized 
information acquisition on the pre-processed data. 

**Model configs**

Since the main components of VAEM are VAEs and PVAE, thus the model configs of VAEM mostly inherit from PVAE (but with 
proper prefixes). For example, in the config files of VAEM, `"marginal_encoder_layers"` stands for the structure of the 
encoder network of marginal VAEs;  `dep_embedding_dim` stands for the dimensionality of embedding of the dependency 
networks. Note however that since the marginal VAEs are vanilla VAEs rather than PVAEs, the configs arguments corresponding 
to set-encoders are disabled.


### 5.3 Predictive VAEM

**Model Description**

In some scenarios, when performing missing data imputation and information acquisition, the user might be having a supervised 
learning problem in mind. That is, the observable variables can be classified into two categories: the input variables 
(covariates), and the output variable (target variable).  Both PVAE and VAEM will treat the input variable and output 
variable (targets) equally, and learns a joint distribution over them. On the contrary, predictive VAEM will simultaneously 
learn a joint distribution over the input variables, as well as a conditional distribution of the target, given the input 
variables. We found that such approach will generally yield better predictive performance on the target variable in practice. 

**The predictive model**

The conditional distribution of the target, given the input variables (as well as the latent representation), is parameterized 
by a feed-forward neural network. This is implemented in [`marginal_vaes_with_predictive_vae`](models/marginal_vaes_with_predictive_vae.py).


**Model configs**

The predictive VAEMs share the same configs as VAEMs.

### 5.4 MNAR Partial VAE

Real-world missing values are often associated with complex generative processes, where the cause of the 
missingness may not be fully observed. This is known as missing not at random (MNAR) data. However, many of the standard 
imputation methods, such as our PVAE and VAEM, do not take into account the missingness mechanism, resulting in biased 
imputation values when MNAR data is present. Also, many practical methods for MNAR does not have identifiability 
guarantees: their parameters can not be uniquely determined by partially observed data, even with access to infinite 
samples. Azua provides a new deep generative model, called MNAR Partial VAE, that addresses both of these issues.

**Mask net and identifiable PVAE**

MNAR PVAE has two main components: a Mask net, and an identifiable PVAE. The mask net is a neural network (implemented 
in [`mask_net`](models/mask_net.py) ), that models the conditional probability distribution of missing mask, given the 
data (and latent representations). This will help debiasing the MNAR mechanism.  The identifiable PVAE is a variant of 
VAE, when combined with the mask net, will provide identifiability guarantees under certain assumptions. Unlike vanilla 
PVAEs, identifiable PVAE uses a neural network, called the prior net, to define the prior distribution on latent space. 
The prior net requires to take some fully observed auxiliary variables as inputs (you may think of it as some side 
information), and generate the distribution on the latent space. By default, unless specified, we will automatically 
treat fully observed variables as auxiliary variables. For more details, please see our paper (link will be available 
in the future).

**Model configs**

Most of the model configs are the same as PVAE, except the following:

* `"mask_net_config"`: This object contains the model configuration of the mask net. 
    - `"decoder_layers"`: The neural network structure of mask net.
    - `"mask_net_coefficient"`: weight of the mask net loss function.
    - `"latent connection"`: if `true`, the mask net will also take as input the latent representations.
    
* `"prior_net_config"`: This object contains the model configuration of the prior net/
    - `"use_prior_net_to_train"`: if `true`, we will use prior net to train the PVAE, instead of the standard normal
    prior.
    - `"encoder_layers"`: the neural network structure of prior net.
    - `"use_prior_net_to_impute"`: use prior net to perform imputation or not. By default, we will always set this to 
    `false`.
    - `"degenerate_prior"`: As mentioned before, we will automatically treat fully observed variables as auxiliary 
    variables. However, in some cases, fully observed variables might not be available (for example, in recommender 
    data). `"degenerate_prior"` will determine how we handle such degenerate case. Currently, we only support `"mask"` 
    method, which will use the missingness mask themselves as auxiliary variables. 


## 6. Other engineering details

**TODO**: Rethink whether we need that section, especially whether we should move supported datasets somewhere

### Reproducibility

**TODO**: Update by bringing up pytorch limitations + link to the source
Running the various Azua scripts give different results on different machines. There has been
some investigation into this, and the team believes this to be due to floating point instability.
Because of this, results should be compared with others from the same machine only.

### Add your own dataset

To add a new dataset, a new directory should be added to the data folder, containing either all of the dataset in a file
named `all.csv`, or a train/test split in files named `train.csv` and `test.csv`. In the former case, a train/test split
will be generated, in a 80%/20% split by default.

Data can be specified in two formats using the `--data_type` flag in the entrypoint scripts. The default format is 
"csv", which assumes that each column represents a feature, and each row represents a data point. The alternative format
is "sparse_csv", where there are 3 columns representing the row ID, column ID and value of a particular matrix element,
as in a coordinate-list (COO) sparse matrix. All values not specified are assumed to be missing.In both cases, no header 
row should be included in the CSV file.

Variable metadata for each variable in a dataset can be specified in an optional file named `variables.json`. This file
is an array of dictionaries, one for each variable in the dataset. For each variable, the following values may be 
specified:

* id: int, index of the variable in the dataset
* query: bool, whether this variable can be queried during active learning (True) or is a target (False).
* type: string, type of variable - either "continuous", "binary" or "categorical".
* lower: numeric, lower bound for the variable.
* upper: numeric, upper bound for the variable.
* name: string, name of the variable.

For each field not specified, it will attempt to be inferred. *Note*: all features will be assumed to be queriable, and
thus not active learning targets, unless explicitly specified otherwise. Lower and upper values will be inferred from
the training data, and the type will be inferred based on whether the variable takes exclusively integer values.

### Supported datasets

Preprocessed datasets and `variables.json` files are available to download from the Azua datasets blob storage
container. Currently supported datasets include:

* UCI datasets: [webpage](https://archive.ics.uci.edu/ml/datasets.php)
  * Bank: [webpage](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
  * Boston Housing: [webpage](http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)
  * Energy: [webpage](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency)
  * Wine: [webpage](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
  * Concrete: [webpage](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)
  * kin8nm: [webpage](https://www.openml.org/d/189)
  * yacht: [webpage](https://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics)
* MNIST: [webpage](http://yann.lecun.com/exdb/mnist/)
* CIFAR-10: [webpage](https://www.cs.toronto.edu/~kriz/cifar.html)

**TODO**: Update the list above

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
