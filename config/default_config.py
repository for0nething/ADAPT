"""
    Parameter dict
    dict of dict
"""
default_args = {
    # power
    "power":{
    "dataset_name" : "power",
    "model_name" : "RQSpline",
    "cuda": 1,
    "non_iid_degree" : 0,
    "data_rng_seed" : 42,

    "IF_DPSGD" : False,

    # Optimization parameters
    "num_epochs" : 120,

    "batch_size" : 512,
    "learning_rate" : 0.0005,

    "momentum" : 0.9,

    # Privacy parameters
    "l2_norm_clip" : 5,
    "noise_multiplier" : 0.7,
    "seed" : 1337,

    "n_client" : 2,


    # Model parameters (In NSF paper)
    "n_layers" : 10,
    "n_hiddens" : [256, 256],
    "n_bins" : 8,

    # Model parameters (Stable)
    "tail_bound" : [-3, 3],
    "grad_norm_clip_value" : 5.0,

    # fed parameters
    "client_steps" : 1000,  # Number of training iterations per round for each client
    "client_epochs": None,  # Priority, steps > epochs
    "check_interval" : 2,   # Global test interval
    "loc_alpha" : 1.0,
    "nsample_clients": 2,
    },

    # imdb
    "imdb": {
        "dataset_name": "imdb",
        "model_name": "RQSpline",
        "cuda": 1,
        "non_iid_degree": 0,
        "data_rng_seed": 42,

        "IF_DPSGD": False,

        # Optimization parameters
        "num_epochs": 120,

        "batch_size": 512,
        "learning_rate": 0.0005,

        "momentum": 0.9,

        # Privacy parameters
        "l2_norm_clip": 5,
        "noise_multiplier": 0.7,
        "seed": 1337,

        "n_client": 2,

        # Model parameters (In NSF paper)
        "n_layers": 10,
        "n_hiddens": [256, 256],
        "n_bins": 8,

        # Model parameters (Stable)
        "tail_bound": [-3, 3],
        "grad_norm_clip_value": 5.0,

        # fed parameters
        "client_steps": 1000,
        "client_epochs": None,
        "check_interval": 2,
        "loc_alpha": 1.0,
        "nsample_clients": 2,
    },


    # flights
    "flights": {
        "dataset_name": "flights",
        "model_name": "RQSpline",
        "cuda": 3,
        "non_iid_degree": 0,
        "data_rng_seed": 42,

        "IF_DPSGD": False,

        # Optimization parameters
        "num_epochs": 120,

        "batch_size": 512,
        "learning_rate": 0.0005,

        "momentum": 0.9,

        # Privacy parameters
        "l2_norm_clip": 5,
        "noise_multiplier": 0.7,
        "seed": 1337,

        "n_client": 2,

        # Model parameters (In NSF paper)
        "n_layers": 10,
        "n_hiddens": [256, 256],
        "n_bins": 8,

        # Model parameters (Stable)
        "tail_bound": [-3, 3],
        "grad_norm_clip_value": 5.0,

        # fed parameters
        "client_steps": 1000,
        "client_epochs": None,
        "check_interval": 2,
        "loc_alpha": 1.0,
        "nsample_clients": 2,
    },


    # BJAQ
    "BJAQ": {
        "dataset_name": "BJAQ",
        "model_name": "RQSpline",
        "cuda": 3,
        "non_iid_degree": 0,
        "data_rng_seed": 42,

        "IF_DPSGD": False,

        # Optimization parameters
        "num_epochs": 120,

        "batch_size": 512,
        "learning_rate": 0.0005,

        "momentum": 0.9,

        # Privacy parameters
        "l2_norm_clip": 5,
        "noise_multiplier": 0.7,
        "seed": 1337,

        "n_client": 2,

        # Model parameters (In NSF paper)
        "n_layers": 10,
        "n_hiddens": [256, 256],
        "n_bins": 8,

        # Model parameters (Stable)
        "tail_bound": [-3, 3],
        "grad_norm_clip_value": 5.0,

        # fed parameters
        "client_steps": 1000,
        "client_epochs": None,
        "check_interval": 2,
        "loc_alpha": 1.0,
        "nsample_clients": 2,  
    },
    # imdbfull
    "imdbfull": {
        "dataset_name": "imdbfull",
        "model_name": "RQSpline",
        "cuda": 3,
        "non_iid_degree": 0,
        "data_rng_seed": 42,

        "IF_DPSGD": False,

        # Optimization parameters
        "num_epochs": 120,

        "batch_size": 512,
        "learning_rate": 0.0005,

        "momentum": 0.9,

        # Privacy parameters
        "l2_norm_clip": 5,
        "noise_multiplier": 0.7,
        "seed": 1337,

        "n_client": 2,

        # Model parameters (In NSF paper)
        "n_layers": 10,
        "n_hiddens": [256, 256],
        "n_bins": 8,

        # Model parameters (Stable)
        "tail_bound": [-3, 3],
        "grad_norm_clip_value": 5.0,

        # fed parameters
        "client_steps": 1000,  
        "client_epochs": None,  
        "check_interval": 2,  
        "loc_alpha": 1.0,
        "nsample_clients": 2,  
    }
}