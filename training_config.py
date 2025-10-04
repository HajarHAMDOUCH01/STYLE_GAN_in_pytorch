training_config = {
    # Architecture
    "image_size": 128,
    "z_dim": 512,
    "w_dim": 512,
    "mapping_layers": 8,  
    
    # Training
    "batch_size": 48,  
    "num_epochs": 300,
    "num_workers": 2,

    "g_lr": 0.002,      
    "d_lr": 0.002,      
    "adam_beta1": 0.0,
    "adam_beta2": 0.99,
    "adam_eps": 1e-8,
    
    "r1_gamma": 10.0,    
    "style_mixing_prob": 0.9,  
    
    "loss_type": "non_saturating",
    
    "r1_interval": 16,  
    "n_critic": 1,  
    
    "start_size": 4,
    "progressive_growing": False,
    
    "dataset_path": "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba",
    "save_dir": "/content/drive/MyDrive/stylegan_checkpoints",
    
    "save_every": 2,
    "sample_every": 2,  
    "log_every": 50,
    
    "grad_clip": None,  
}