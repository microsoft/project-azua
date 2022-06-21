# Project Azua

## 0. Overview

Humans make tens of thousands of decisions every day. Project Azua aims to develop machine learning solutions for efficient decision making that demonstrate human expert-level performance across all domains. Our conceptual framework is to divide decisions into two types: "best next question" and "best next action". 

For **Causal ML** and **DECI** see the new [repo](https://github.com/microsoft/causica)

In daily life, one type of decision we make relates to information gathering for "get to know" decisions; for example, a medical doctor takes a medical test to decide the correct diagnosis for a patient. Humans are very efficient at gathering information and drawing the correct conclusion, while most deep learning methods require significant amounts of training data. Thus, the first part of project Azua focuses on enabling machine learning solutions to gather personalized information, allowing the machine to know the "best next question" and make a final judgment efficiently [1,2,6].
Our technology for "best next question" decisions is driven by state-of-the-art algorithms for Bayesian experimental design and active learning.

With these decision-making goals, one can use our codebase in an end-to-end way for decision-making. We also provide the flexibility to use any core functionalities such as missing value prediction, best next question, etc, separately depending on the users' needs.     

Our technology has enabled personalized decision-making in real-world systems, combining multiple advanced research methodologies in simple APIs suitable 
for research development in the research community, and commercial use by data scientists and developers. For commercial applications, please reach out to us at  azua-request@microsoft.com  if you are interested in using our technology as a service. 


### References

If you have used the models in our code base, please consider to cite the corresponding paper:

[1], **(PVAE and information acquisition)** Chao Ma, Sebastian Tschiatschek, Konstantina Palla, Jose Miguel Hernandez-Lobato, Sebastian Nowozin, and Cheng Zhang. "EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE." In International Conference on Machine Learning, pp. 4234-4243. PMLR, 2019.

[2], **(VAEM)** Chao Ma, Sebastian Tschiatschek, Richard Turner, José Miguel Hernández-Lobato, and Cheng Zhang. "VAEM: a Deep Generative Model for Heterogeneous Mixed Type Data." Advances in Neural Information Processing Systems 33 (2020).

[3], **(Eedi dataset)** Zichao Wang, Angus Lamb, Evgeny Saveliev, Pashmina Cameron, Yordan Zaykov, Jose Miguel Hernandez-Lobato, Richard E. Turner et al. "Results and Insights from Diagnostic Questions: The NeurIPS 2020 Education Challenge." arXiv preprint arXiv:2104.04034 (2021).

[4], **(CORGI:)** Jooyeon Kim, Angus Lamb, Simon Woodhead, Simon Pyton Jones, Cheng Zhang, and Miltiadis Allamanis. CORGI: Content-Rich Graph Neural Networks with Attention. In GReS: Workshop on Graph Neural Networks for Recommendation and Search, 2021

[5], **(GINA)** Chao Ma and Cheng Zhang. Identifiable Generative Models for Missing Not at Random Data Imputation. In Advances in Neural Information Processing Systems 34 (2021)

[6], **(Transformer PVAE, Transformer encoder PVAE, Rupert)** Sarah Lewis, Tatiana Matejovicova, Angus Lamb, Yordan Zaykov, 
Miltiadis Allamanis, and Cheng Zhang. Accurate Imputation and Efficient Data Acquisition with 
Transformer-based VAEs.  In NeurIPS: Workshop on Deep Generative Models and Applications (2021)


### Resources

For quick introduction to our work regarding best next question, checkout [our NeurIPS 2020 tutorial, from 2:17:11](https://slideslive.com/38943571/advances-in-approximate-inference).
For a more in-depth technical introduction of deep genertive model for missing value prediction and best next question, checkout [our ICML 2020 tutorial](https://slideslive.com/38931299/efficient-missingvalue-acquisition-with-variational-autoencoders?ref=account-folder-55866-folders)

## 1. Core functionalities

Azua has there core functionalities as shown below depends on the type of decision maksing tasks.

![AZUA](Azua%20Image.jpg)

### 1.1. Missing Value Prediction (MVP)

In many real-life scenarios, we will need to make decisions under incomplete information. Therefore, it is crucial to 
make accurate estimates regarding the missing information. To this end, Azua provides state-of-the-art missing
value prediction methods based on deep learning algorithms. These methods are able to learn from incomplete data, and then 
perform missing value imputation. Instead of producing only one single values for missing entries
as in common softwares, most of our methods are able to return multiple imputation values, which provide valuable 
information of imputation uncertainty. We work with data with same type [1],  as well as mixed type data and different 
missing patterns [2]. 

### 1.2. Personalized active information acquisition/Best next question (BNQ)

Azua can not only be used as a powerful data imputation tool, but also as an engine
to actively acquire valuable information [1]. Given an incomplete data instance, Azua is able to suggest which unobserved 
variable is the most informative one (subject to the specific data instance and the task) to collect, using 
information-theoretic approaches. This allows the users to focus on collecting only the most important information, 
and thus make decisions with minimum amount of data. 

Our active information acquisition functionality has two modes: i) if there is a specific variable (called target 
variable) that the user wants to predict, then Azua will suggest the next variable to collect, that is most valuable to 
predicting that particular target variable. ii) otherwise, Azua will make decisions using built-in criterion.   

If you are interested in using service API of MVP or NBQ, please read [here](https://techcommunity.microsoft.com/t5/ai-customer-engineering-team/using-ai-to-know-which-question-to-ask-and-when-to-ask-it/ba-p/2768799)

### 1.3 Causal discovery under missing data (CD)

The underlying causal relationships behind data crucial for real-world decision making. However, discovering causal
structures under incomplete information is difficult. Azua provide state-of-the-art solution based on graph neural nets,
 which is able to tackle missing value imputation and causal discovery problems at the same time.


## 2. Getting started

### Set-up Python environement

A [conda environment](environment.yml) is used to manage system requirements. To install conda, check the
installation instructions [here](https://docs.conda.io/en/latest/miniconda.html).
To create the azua environment, run

```bash
conda env create -f environment.yml
```

And to activate the environment run

```bash
conda activate azua
```

### Download dataset

You need to download the dataset you want to use for your experiment, and put it under relevant *data/*'s subdirectory e.g. putting *yahoo* dataset under *data/yahoo* directory. For the list of the supported datasets, please refer to [the Supported datasets section](#supported-datasets).

For some of the UCI dataset, you can use *download_dataset.py* script for downloading the dataset e.g.:
```bash
python download_dataset.py boston
```

### Run experiment 

[`run_experiment.py`](run_experiment.py) script runs any combination of model training, imputation and active learning. An example of running experiment is:

```bash
python run_experiment.py boston -mt pvae -i -a eddi rand
```

In this example, we train a PVAE model (i.e. "*-mt*" parameter) on the Boston Housing dataset (i.e. first parameer), evaluate the imputation performance on the test set (i.e. "*-i*" parameter) and compare the sequential feature acquisition performance between the EDDI policy and random policy (i.e. "*-a*" parameter). For more information on running experiments, available parameters etc., please run the following command:

```bash
python run_experiment.py --help
```

We also provide more examples of running different experiments in the section below.

## 3. Model overview

Below we summarize the list of models currently available in Azua, their descriptions, functionalities (**MVP = missing 
value prediction, NBQ = personalized information acquisition/next best quesiton, CD = Causal discovery**), and an example code that shows how 
to run the model (which will also reproduce one experiment from the paper). 

Model | Description | Functionalities | Example usage 
--- | --- | --- | --- 
[Partial VAE (PVAE)](azua/models/partial_vae.py)                                | An extension of VAEs for <br /> partially observed data.  <br /> See [our paper](http://proceedings.mlr.press/v97/ma19c.html). |  MVP, BNQ | `python run_experiment.py boston -mt pvae -a eddi rand`
[VAE Mixed (VAEM)](azua/models/vae_mixed.py)                                    | An extension of PVAE for <br /> heterogeneous mixed type data.  <br /> See [our paper](https://papers.nips.cc/paper/2020/hash/8171ac2c5544a5cb54ac0f38bf477af4-Abstract.html). | MVP, BNQ | `python run_experiment.py bank -mt vaem_predictive -a sing` 
[MNAR Partial VAE (MNAR-PVAE)](azua/models/mnar_pvae.py)                        | An extension of VAE that <br /> handles missing-not-at-random  <br /> (MNAR) data.  <br /> More details in the future. | MVP, BNQ | `python run_experiment.py yahoo -mt mnar_pvae -i` 
[Bayesian Partial VAE (B-PVAE)](azua/models/bayesian_pvae.py)                   | PVAE with a Bayesian treatment. | MVP, BNQ | `python run_experiment.py boston -mt bayesian_pvae -a eddi rand`
[Transformer PVAE](azua/models/transformer_pvae.py)                             | A PVAE in which the encoder <br />  and decoder contain transformers.  <br /> See [our paper](https://openreview.net/pdf?id=N_OwBEYTcKK)| MVP, BNQ | `python run_experiment.py boston -mt transformer_pvae -a eddi rand`
[Transformer encoder PVAE](azua/models/transformer_encoder_pvae.py)             | A PVAE in which the encoder <br /> contains a transformer.  <br /> See [our paper](https://openreview.net/pdf?id=N_OwBEYTcKK) | MVP, BNQ | `python run_experiment.py boston -mt transformer_encoder_pvae -a eddi rand`
[Transformer imputer/Rupert](azua/models/transformer_imputer.py)                | A simple transformer model. <br /> See [our paper](https://openreview.net/pdf?id=N_OwBEYTcKK) | MVP, BNQ | `python run_experiment.py boston -mt transformer_imputer -a variance rand`
[CoRGi](azua/models/corgi.py)                                                   |  GNN-based imputation with <br /> emphasis on item-related data  <br /> based on [Kim et al.](https://toappear) | MVP | See 5.7.1-5.7.4 for details. 
[Graph Convolutional Network (GCN)](azua/models/graph_convolutional_network.py) |  GNN-based imputation based <br /> on [Kipf et al.](https://arxiv.org/abs/1609.02907) | MVP | See *5.7.2-5.7.4* for details. 
[GRAPE](azua/models/grape.py)                                                   |  GNN-based imputation based <br /> on [You et al.](https://snap.stanford.edu/grape/) | MVP | See *5.7.2-5.7.4* for details. 
[Graph Convolutional Matrix Completion (GC-MC)](azua/models/gcmc.py)            |   GNN-based imputation based <br /> on [van den Berg et al.](https://arxiv.org/abs/1706.02263) | MVP | See *5.7.2-5.7.4* for details. 
[GraphSAGE](azua/models/graphsage.py)                                           |  GNN-based imputation based  <br /> on [Hamilton et al.](https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html) | MVP | See [5.7.2-5.7.4](####5.7.2 Different node initializations) for details. 
[Graph Attention Network (GAT)](azua/models/graph_attention_network.py)         | Attention-based GNN imputation  <br /> based on [Veličković et al.](https://arxiv.org/abs/1710.10903) | MVP | See [5.7.2-5.7.4](####5.7.2 Different node initializations) for details. 
[Deep Matrix Factorization (DMF)](azua/models/deep_matrix_factorization.py)     | Matrix factorization with NN architecture. See [deep matrix factorization](https://www.ijcai.org/Proceedings/2017/0447.pdf.) | MVP | `python run_experiment.py eedi_task_3_4_binary -mt deep_matrix_factorization`
[Mean imputing](azua/baselines/mean_imputing.py)                                | Replace missing value with  <br /> mean. | MVP | `python run_experiment.py boston -mt mean_imputing`
[Zero imputing](azua/baselines/zero_imputing.py)                                | Replace missing value with  <br /> zeros. | MVP | `python run_experiment.py boston -mt zero_imputing`
[Min imputing](azua/baselines/min_imputing.py)                                  | Replace missing value with  <br /> min value. | MVP | `python run_experiment.py boston -mt min_imputing`
[Majority vote](azua/baselines/majority_vote.py)                                | Replace missing value with  <br /> majority vote. | MVP | `python run_experiment.py boston -mt majority_vote`
[MICE](azua/baselines/mice.py)                                                  | Multiple Imputation by  <br /> Chained Equations,  <br /> see [this paper](https://onlinelibrary.wiley.com/doi/full/10.1002/sim.4067) | MVP | `python run_experiment.py boston -mt mice`
[MissForest](azua/baselines/missforest.py)                                      | An iterative imputation method  <br /> (missForest) based on random forests.  <br /> See [this paper](https://academic.oup.com/bioinformatics/article/28/1/112/219101) | MVP | `python run_experiment.py boston -mt missforest` 

## Objectives

Next Best Question Objectives | Description
--- | ---
[EDDI](azua/objectives/eddi.py) | It uses information gain given observed values to predict the next best feature to query.
[SING](azua/objectives/sing.py) | It uses a fixed information gain ordering based on no questions being asked.
[Random](azua/objectives/rand.py) | It randomly selects the next best feature to query.
[Variance](azua/objectives/variance.py) | It queries the feature that is expected to reduce predictive variance in the target variable the most.

## 4. Reference results

### Supported datasets

We provide `variables.json` files and model configurations for the following datasets:

* UCI datasets: [webpage](https://archive.ics.uci.edu/ml/datasets.php)
  * Bank: [webpage](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
  * Boston Housing: [webpage](http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)
  * Concrete: [webpage](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)
  * Energy: [webpage](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency)
  * Iris: [webpage](https://archive.ics.uci.edu/ml/datasets/iris)
  * Kin8nm: [webpage](https://www.openml.org/d/189)
  * Wine: [webpage](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
  * Yacht: [webpage](https://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics)
* MNIST: [webpage](http://yann.lecun.com/exdb/mnist/)
* CIFAR-10: [webpage](https://www.cs.toronto.edu/~kriz/cifar.html)
* NeurIPS 2020 Education Challenge datasets: [webpage](https://eedi.com/projects/neurips-education-challenge)
  * eedi_task_1_2_binary: The data for the first two tasks. It uses only correct (1) or wrong (0) answers.
  * eedi_task_1_2_categorical: The data for the first two tasks. It uses A, B, C, D answers.
  * eedi_task_3_4_binary: The data for the last two tasks. It uses only correct(1) or wrong (0) answers.
  * eedi_task_3_4_categorical: The data for the last two tasks. It uses A, B, C, D answers.
  * eedi_task_3_4_topics: The data for the last two tasks. To produce the experimental results in VISL, binary answers are used. It has additional topic metadata.
* Neuropathic Pain Diagnosis Simulator Dataset: [webpage](https://github.com/TURuibo/Neuropathic-Pain-Diagnosis-Simulator)
  * denoted as "Neuropathic_pain" below. You need to use the simulator to generate the data.
* Yahoo [webpage](https://consent.yahoo.com/v2/collectConsent?sessionId=3_cc-session_f8d7b45f-c09b-473f-99c6-d27adf00f176)
* Goodreads [webpage](https://github.com/bahramJannesar/GoodreadsBookDataset): Refer to section 5.7.3 for more details.

### Missing Value Prediction (MVP)

#### Test Data Normaliized RMSE

For evalaution, we apply row-wise splitting, and we use 30% holdout data to test.

Dataset | Partial <br /> VAE | VAEM | Predictive <br /> VAEM | MNAR <br /> Partial <br /> VAE | B-PVAE | Mean <br /> imputing | Zero <br /> imputing | Min <br /> imputing | Majority <br /> vote | MICE | MissForest
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
Bank | 0.51 |  0.66 | 0.56 | --| --| -- | -- | -- | 0.51 | -- | --
Boston | 0.17 | 0.18 | -- | -- | 0.18| 0.23 | -- | -- | 0.37 | -- | 0.15
Conrete | 0.18 | 0.19 | -- | -- | --| 0.22 | -- | -- | 0.27 | -- | 0.13
Energy | 0.22 | 0.32 | -- | -- | 0.25| 0.35 | -- | -- | 0.48 | -- | 0.24
Iris | 0.59 | -- | -- | -- | --| -- | -- | -- | -- | -- | --
Kin8nm | 0.27 | -- | -- | -- | --| -- | -- | -- | -- | -- | --
Wine | 0.17 | 0.17 | -- | -- | --| 0.24 | -- | -- | 0.31 | -- | 0.17 
Yacht | 0.24 | 0.23 | -- | -- | --| 0.3 | -- | -- | 0.3 | -- | 0.16
Yahoo | 0.36 | -- | -- | 0.3 | --| -- | -- | -- | -- | -- | --

#### Accuracy

Please note that for binary data (e.g. eedi_task_3_4_binary), we report accuracy to compare with the literature.

Dataset | Partial <br /> VAE | VISL | CORGI | GRAPE | GCMC| Graph <br /> Convolutional <br /> Network | Graph <br /> Attention <br /> Network | GRAPHSAVE
--- | --- | --- | --- | --- | --- | --- | --- | ---
eedi_task_3_4_binary | 0.72 | -- | 0.71 | 0.71 | 0.69 | 0.71 | 0.6 | 0.69
eedi_task_3_4_categorical | 0.57 | -- | -- | -- | -- | -- | -- | --
eedi_task_3_4_topics | 0.71 | 0.69 | -- | -- | -- | -- | -- | --
Neuropathic_pain | 0.94 | 0.95 | -- | -- | -- | -- | -- | --

### Next Best Question (NBQ): area under information curve (AUIC)

To evaluate the performance of different models for NBQ task, we compare the area under information curve (AUIC). See 
[our paper](https://papers.nips.cc/paper/2020/hash/8171ac2c5544a5cb54ac0f38bf477af4-Abstract.html) for details. AUIC 
is calculated as follows: at each step of the NBQ, each model will propose to collect one variable, and make new predictions 
 for the target variable. We can then calculate the predictive error (e.g., rmse) of the target variable at each step. 
 This creates the information curve as the NBQ task progresses. Therefore, the area under the information
curve (AUIC) can then be used to compare the performance across models and strategies. Smaller AUIC value indicates 
better performance. 

Dataset | Partial <br /> VAE | VAEM | Predictive <br /> VAEM | MNAR <br /> Partial <br /> VAE | B-PVAE
--- | --- | --- | --- | --- | ---
Bank | 6.6 |  6.49 | 5.91 | --| --
Boston | 2.03 | 2.0 | -- | -- | 1.96
Conrete | 1.48 | 1.47 | -- | -- | --
Energy | 1.18 | 1.3 | -- | -- | 1.44
Iris | 2.8 | -- | -- | -- | --
Kin8nm | 1.28 | -- | -- | -- | --
Wine | 2.14 | 2.45 | -- | -- | --
Yacht | 0.94 | 0.9 | -- | -- | --

## 5. Model details

### 5.1 Partial VAE

**Model Description**

Partial VAE (PVAE) is an unsupervised deep generative model, that is specifically designed to handle missing data. We mainly
use this model to learn the underlying structure (latent representation) of the partially observed data, and perform missing data
imputation. Just like any vanilla VAEs, PVAE is comprised of an encoder and a decoder. The PVAE encoder is parameterized 
by the so-called set-encoder (point-net, see [our paper](http://proceedings.mlr.press/v97/ma19c.html) for details), which is able 
to extract the latent representation from partially observed data. Then, the PVAE decoder can take as input the extracted 
latent representation, and generate values for both missing entries (imputation), and observed entries (reconstruction).  

**The partial encoder**

One of the major differences between PVAE and VAE is, the PVAE encoder can handle missing data in a principled way. The 
PVAE encoder is parameterized by the so-called set-encoder, which will process partially observed data in three steps: 
1, feature embedding; 2, permutation-invariant aggregation; and 3, encoding into statistics of latent representation. 
These are implemented in [`feature_embedder.py`](azua/models/feature_embedder.py), ['point_net.py'](azua/models/point_net.py), and 
[`encoder.py`](azua/models/encoder.py), respectively. see [our paper, Section 3.2](http://proceedings.mlr.press/v97/ma19c.html) for technical details. 


**Model configs**

* `"embedding_dim"`: dimensionality of embedding (referred to as **e** in the paper) for each input to PVAE encoder. 
    See [our paper](http://proceedings.mlr.press/v97/ma19c.html) for details.
* `"set_embedding_dim"`: dimensionality of output set embedding (referred to as **h** in the paper) in PVAE encoder.
    See [our paper](http://proceedings.mlr.press/v97/ma19c.html) for details.
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
[`marginal_vaes.py`](azua/models/marginal_vaes.py). Those one-dimensional VAEs will capture the marginal properties of each 
variable and provide a latent representation that is more homogeneous across dimensions. 

* In the second stage, we capture  the dependencies among each variables. To this end, another Partial VAE, called the 
dependency network, is build on top of the latent representations provided by the first-stage VAEs. This is implemented 
in [`dependency_network_creator`](azua/models/dependency_network_creator.py)

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
by a feed-forward neural network. This is implemented in [`marginal_vaes_with_predictive_vae`](azua/models/marginal_vaes_with_predictive_vae.py).


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
in [`mask_net`](azua/models/mask_net.py) ), that models the conditional probability distribution of missing mask, given the 
data (and latent representations). This will help debiasing the MNAR mechanism.  The identifiable PVAE is a variant of 
VAE, when combined with the mask net, will provide identifiability guarantees under certain assumptions. Unlike vanilla 
PVAEs, identifiable PVAE uses a neural network, called the prior net, to define the prior distribution on latent space. 
The prior net requires to take some fully observed auxiliary variables as inputs (you may think of it as some side 
information), and generate the distribution on the latent space. By default, unless specified, we will automatically 
treat fully observed variables as auxiliary variables.

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


**Data processing for Yahoo dataset**

To download Yahoo dataset, you need to apply first here via this webpage [webpage.](https://consent.yahoo.com/v2/collectConsent?sessionId=3_cc-session_f8d7b45f-c09b-473f-99c6-d27adf00f176) 
Once you have received the data, simply unzip it and you will get  `ydata-ymusic-rating-study-v1_0-train.txt` and `ydata-ymusic-rating-study-v1_0-test.txt`.
Since the train/test split in files are already provided, we only need to convert the two `.txt` files to `.csv` format
(simply copy and paste the data in two `.txt` file to `train.csv` and `test.csv` respectively, and put them under `data/yahoo/`). 
This resulting dataset contains an MNAR training set of more than 300K self-selected ratings from 15,400 users on 1,000 
songs, and an MCAR test set of randomly selected ratings from 5,400 users on 10 random songs. Currently, we treat all the 
ratings as continuous variables. This can also be configured in `variables.json` by specifying the `"type"` field, which 
is detailed in Section 6. Then, we can run the model by `python run_experiment.py yahoo -mt mnar_pvae -i`.

**Handling auxiliary variables**

As mentioned before, this model requires some fully observed auxiliary variables as inputs. In most cases, this is automatically 
handled by our implementation: whenever there are columns that are fully observed across all data instances (i.e., 
`"always_observed" = true` in the `variables.json` file), we will simply treat those columns as auxiliary variables. When all columns have at least 
one entry missing, we will just take the missingness mask indicator as the auxiliary variables. It is also possible to 
manually specify which variables are auxiliary, by simply editing the `variables.json` file. This can be done by specifying 
the corresponding auxiliary variables under `"auxiliary_variables"` in `variables.json`. In the example below, we have 
created a `variables.json` file with `var_1` being an auxiliary variable, and `var_2`, `var_3` being other normal variables 
that we need to model.
```
{
    "variables": [
        {
            "id": 2,
            "lower": 0.0,
            "name": "var_2",
            "query": true,
            "type": "continuous",
            "upper": 10.0
        },
        {
            "id": 3,
            "lower": 0.0,
            "name": "var_3",
            "query": false,
            "type": "continuous",
            "upper": 10.0
        }
    ],
    "auxiliary_variables": [
        {
            "id": 1,
            "lower": 0.0
            "name": "var_1",
            "query": true,
            "type": "continuous",
            "upper": 10.0
        }
    ]
}
```



### 5.5 Bayesian partial VAE (B-PVAE)

Standard training of PVAE produces the point estimates for the neural network parameters in the decoder. This approach 
does not quantify the epistemic uncertainty of our model. B-PVAE is a variant of PVAE, that applies a fully Bayesian 
treatment to the weights. The model setting is the same as in [BELGAM,](https://proceedings.neurips.cc/paper/2019/file/c055dcc749c2632fd4dd806301f05ba6-Paper.pdf) 
whereas the approximate inference is done using the [inducing weights approach](https://arxiv.org/abs/2105.14594). 

**Implementation**

Implementation-wise, B-PVAE is based on Bayesianize, a lightweight Bayesian neural network (BNN) wrapper in pytorch, 
which allows easy conversion of neural networks in existing scripts to its Bayesian version with minimal changes. For 
more details, please see our [github repo](https://github.com/microsoft/bayesianize).

### 5.6 CoRGi, Graph Convolutional Network (GCN), GRAPE, Graph Convolutional Matrix Completion (GC-MC), and GraphSAGE

#### 5.6.1 CoRGi and baselines

**[CoRGi](azua/models/corgi.py)** 

CoRGi is a GNN model that considers the rich data within nodes in the context of their neighbors. 
This is achieved by endowing CORGI’s message passing with a personalized attention
mechanism over the content of each node. This way, CORGI assigns user-item-specific 
attention scores with respect to the words that appear in items. More detailed information can be found in our paper:

CORGI: Content-Rich Graph Neural Networks with Attention. J. Kim, A. Lamb, S. Woodhead, S. Peyton Jones, C. Zhang, 
M. Allamanis. RecSys: Workshop on Graph Neural Networks for Recommendation and Search, 2021, 2021

**[Graph Convolutional Network (GCN)](azua/models/graph_convolutional_network.py)**


Azua provides a re-implementation of GCN. As a default, "average" is used for the aggregation function 
and nodes are randomly initialized. 
We adopt dropout with probability 0.5 for node embedding updates
as well as for the prediction MLPs.

**[GRAPE](azua/models/grape.py)**

GRAPE is a GNN model that employs edge embeddings (please refer to [this paper](https://arxiv.org/abs/2010.16418) for details).
Also, it adopts edge dropouts that are applied throughout all message-passing layers.
Compared to the GRAPE proposed in the oroginal paper, because of the memory issue, 
we do not initialize nodes with one-hot vectors nor constants (ones). 

**[Graph Convolutional Matrix Completion (GC-MC)](azua/models/gcmc.py)**

Compared to GCN, this model has a single message-passing layer. 
Also, For classification, each label is endowed with a separate message passing channel.
Here, we do not implement the weight sharing. For more details, please refer to [this paper](https://arxiv.org/abs/1706.02263). 

**[GraphSAGE](azua/models/graphsage.py)**

GraphSAGE extends GCN by allowing the model to be trained on the part of the graph, 
making the model to be used in inductive settings. For more details, please refer to [this paper](https://arxiv.org/abs/1706.02216)

**[Graph Attention Network (GAT)](azua/models/graph_attention_network.py)**

During message aggregation, GAT uses the attention mechanism to allow the target nodes to
distinguish the weights of multiple messages from the source nodes for aggregation. For more details, please refer to 
[this paper](https://arxiv.org/abs/1710.10903). 

#### 5.6.2 Different node initializations

All GNN models allow different kinds of node initializations.
This can be done by modifying the model config file.
For example, to run CoRGi with SBERT initialization, change `"node_init": "random"`
to `"node_init": "sbert_init"` in `configs/defaults/model_config_corgi.json`.

The list of node initializations allowed inclue: 

`"random", "grape", "text_init" (TF-IDF),"sbert_init", "neural_bow_init", "bert_cls_init", "bert_average_init"`

For example, the test performance of `GCN Init: NeuralBOW` in Table 2 of the paper on Eedi dataset can be acquired by running:

`python run_experiment.py eedi graph_convolutional_network -dc configs/defaults/model_config_graph_convolutional_network.json`

with `"node_init": "neural_bow_init"` in te corresponding model config file.

#### 5.6.3 Datasets

CoRGi operate on content-augmented graph data.

**Goodreads**

Download the data from this [link](https://github.com/bahramJannesar/GoodreadsBookDataset) under `data` directory with name `goodreads`.

The Goodreads dataset from the Goodreads website contains users and books. The content of each book-node is its natural language description. The dataset includes a 1 to 5 integer ratings between some books and users.

The pre-processing of this data can found at

```
research_experiments/GNN/create_goodreads_dataset.py
```

**Eedi**

Download the data from this [link](https://competitions.codalab.org/competitions/25449) under `data` directory with name `eedi`.

This dataset is from the Diagnostic Questions - NeurIPS 2020 Education Challenge. It contains anonymized student and question identities with the student responses to some questions. The content of each question-node is the text of the question. Edge labels are binary: one and zero for correct and incorrect answers.

The pre-processing codes for the datasets to be used for CoRGi can be found at:

```
research_experiments/eedi/
```

#### 5.6.4 Running Corgi

To run the CoRGi code with Eedi dataset, first locate the preprocessed data at

```
data/eedi/
```

Then, run the following code:

```
python run_experiment.py eedi -mt corgi
```

This can be done with different datasets and different GNN models. The train and validation performances can be tracked 
using tensorboard which is logged under the `runs` directory. Also, the trained model is saved with `.pt` extension.

## 6. Other engineering details

### Reproducibility

As the project uses PyTorch, we can't guarantee completely reproducible results across different platforms and devices. However, for the specific platform/device, the results should be completed reproducible i.e. running an experiment twice should give the exact same results. More about limitation on reproducibility in PyTorch can be found [here](https://pytorch.org/docs/stable/notes/randomness.html).

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

* always_observed: bool, whether this variable is always observed across all data instances.
* id: int, index of the variable in the dataset
* query: bool, whether this variable can be queried during active learning (True) or is a target (False).
* type: string, type of variable - either "continuous", "binary" or "categorical".
* lower: numeric, lower bound for the variable.
* upper: numeric, upper bound for the variable.
* name: string, name of the variable.

For each field not specified, it will attempt to be inferred. *Note*: all features will be assumed to be queriable, and
thus not active learning targets, unless explicitly specified otherwise. Lower and upper values will be inferred from
the training data, and the type will be inferred based on whether the variable takes exclusively integer values.

### Split type for the dataset

The source data can be split into train/validation/test datasets either based on rows or elements. The former is split by rows of the matrix, whereas the latter is split by individual elements of the matrix, so that different elements of a row can appear in different data splits (i.e. train or validation or test).

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
