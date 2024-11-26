# ADAPT
The training and evaluation code as well as the datasets used in our ADAPT paper. 

# Quick Start

## Folder Structure

    .
    ├── config                  # Parameter Settings for each dataset
    ├── datasets                # Dataset related code
    ├── dependency              # Dependent libraries, including modified versions of nflows and torchquad
    ├── evaluate                # Evaluation code, conducting data analytics using trained models    
    ├── DP                      # DP related code
    ├── FL                      # FL related code
    ├── model                   # Model related code
    ├── nets                    # Components of the models
    ├── transform               # Components of the NF model
    ├── environment.yml         # Environment file
    ├── TR.pdf                  # The technical report of our paper
    ├── my_tree_util.py         # Utility functions
    ├── train_ADAPT.py          # Train code for ADAPT
    ├── train_central.py        # Train code for centralized training
    ├── train_fed.py            # Train code for conventional FL training
    └── tran_to_eval.py         # Process the model before evaluation
             
## Requirements

Our implementation is mainly based on JAX, torch and a set of related python packages. 
The full list of dependencies can be found at [`environment.yml`](environment.yml).
Some of the important dependencies are: 
- `JAX==0.3.25`
- `jaxlib==0.3.25`
- `distrax==0.1.3`
- `optax==0.1.5`
- `flax==0.6.1`
- [`nflows (modified version)`](dependency/nflows)
- [`torchquad (modified version)`](dependency/torchquadMy)



## Usage
The real-world datasets can be downloaded from [link](https://drive.google.com/drive/folders/1tVBbGWdoEG5MRkf8trXIc5pmtpyri5OU?usp=sharing).
- **Step 1:** Build conda environment with `conda env create -f environment.yml`.
- **Step 2:** Switch to the installed environment by `conda activate jax`.
- **Step 3:** Install modified torchquad by `cd ./torchquadMy`, and then `pip install .` .
- **Step 4:** Install modified nflows by `cd ./nflows`, and then `pip install .` .
- **Step 5:** Train density models using `train_ADAPT.py`.
- **Step 6:** Transform the model for evaluation using `tran_to_eval.py`.
- **Step 7:** Evaluate the trained model by different analytical queries using `./evaluate/evaluate.py`.


## Detailed Parameter Description

### Train
Before training using [`train_ADAPT.py`](train_ADAPT.py), make sure the experiment parameters are set correctly. The main experimental parameters include:
- **Parameters:**
  - **dataset_name:** Dataset to use. The available datasets include `power`, `BJAQ`, `imdbfull`, and `flights`.
  - **model_name:** The network structure to use. `RQSpline` denotes the NF model with complete fully connected layers. `MaskRQSpline` denotes the NF model with structural different neurons proposed in the paper. 
  - **n_client:** Number of clients.
  - **num_epochs:** Number of rounds in federated training.
  - **loc_alpha:** Proportion of removed neural connections, i.e., $\rho$ in the paper.
  - **client_steps:** Number of local steps in local training.
  - **client_epochs:** Number of epochs in local training. Only need to assign one of `client_steps` and `client_epochs`.
  - **learning_rate:** Learning rate in local training.
  - **sup_learning_rate:** Learning rate of the fine-tuning based on the distribution average proposed in the paper.  
  - **sup_itr_num:** Number of iterations in the fine-tuning based on the distribution average proposed in the paper.
  - **l2_norm_clip:** L2 norm clipping in DP training.
  - **noise_multiplier:** Noise multiplier in DP training.
  - **n_layers:** Number of layers in each coupling layer
  - **n_hiddens:** Number of hidden neurons in each hidden layer. It should be a list with length `n_layers`.



### Evaluate
Before evaluation using [`evaluate.py`](evaluate/evaluate.py), make sure the experiment parameters are set correctly. For evaluation, these parameters are set in [`parameterSetting.py`](evaluate/parameterSetting.py).The main experimental parameters include:
- **Parameters:**
  - **dataset_name:** Dataset to use. The available datasets include `power`, `BJAQ`, `imdbfull`, and `flights`.
  - **numQuery:** Number of queries for evaluation.
  - **agg_type:** The category of aggregate queries for evaluation. The available chocies include `count`, `sum`, `average`, `variance`, `mode`, `percentile` and `range`.
  - **PERCENT:** Only usable when `agg_type` is `percentile`, denoting the target percentile.
  - **ERROR_METRIC:** Error metric for accuracy evaluation of the estimation. Default `relative`.

  
### Baseline Methods
We also include a set of baselines within our proposed federated data analytics paradigm or other conventional paradigms: 
- **CentralDP** and **Central:** Centralized training with and without DP.
  - Use [`train_central.py`](train_central.py). The bool parameter `IF_DPSGD` controls whether using DP training, i.e. CentralDP or Central. Other hyper-parameters are generally the same as those for ADAPT.
- **FedAVG:** the classic federated learning algorithm that adopts coordinate-wise averaging to fuse local models. 
  - Use [`train_fed.py`](train_fed.py). Set `model_name` as `RQSpline`. Other hyper-parameters are generally the same as those for ADAPT.
- **FedPAN:** pre-aligns neurons by perturbing the neuron outputs with a location-based periodic function. 
  - Use [`train_fed.py`](train_fed.py). Set `model_name` as `PosAddSpline` or `PosMulSpline`. Set `loc_alpha` to control the weight of periodic function. Other hyper-parameters are generally the same as those for ADAPT.


- **FedMA:**
  - We used the official source [code](https://github.com/IBM/FedMA). The NF model with the same hyperparameters as ADAPT is trained and then used for evaluation.
- **GAMF:**
  - We used the official source [code](https://github.com/Thinklab-SJTU/GAMF). The NF model with the same hyperparameters as ADAPT is trained and then used for evaluation.



- **DPSQL:**
  - We used the official source [code](https://github.com/google/differential-privacy). It protects data privacy by adding noise to query results. A privacy budget is evenly allocated across all queries.
- **FLEX:**
  - We used the official source [code](https://github.com/uber-archive/sql-differential-privacy). It is also a method based on output-perturbation. We assign the same privacy budgets as DPSQL to each query.
- **PrivMRF:**
  - We used the official source [code](https://github.com/caicre/PrivMRF). The synthesized data with the same size as the original data is generated under the same $(\epsilon, \delta)$-DP guarantee using the MRF model. Then the query is executed on the synthesized data.
- **DP-WGAN:**
  - We used the official source [code](https://github.com/nesl/nist_differential_privacy_synthetic_data_challenge). The synthesized data with the same size as the original data is generated under the same $(\epsilon, \delta)$-DP guarantee using the WGAN model. Then the query is executed on the synthesized data.



### Test Environment
All experiments are performed on a Ubuntu server with an Intel(R) Xeon(R) 6242R CPU and a Nvidia 3090 GPU.

### License
Please note that this repo is the code of an academic paper under reveiw. We have not yet determined the license. So we beg for not using this code for other purposes at this time. Thanks for understanding.

