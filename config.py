import os

config = {
    'ngf': 32,                # Number of filters in the generator
    'n_blocks': 9,            # Number of residual blocks
    'ndf': 64,                # Number of filters in the discriminator
    'lr': 0.0002,             # Learning rate
    'batch_size': 8,          # Batch size
    'img_size': 256,          # Image size (resize)
    'identity_weight': 5.0,   # Identity loss weight
    'cycle_weight': 10.0,     # Cycle loss weight
    'adv_weight': 1.0,        # Adversarial loss weight
    'trainYoung_dir': '/path/to/trainYoung',  # Path to the young age dataset
    'trainMiddle_dir': '/path/to/trainMiddle',  # Path to the middle-aged dataset
    'trainOld_dir': '/path/to/trainOld',  # Path to the old age dataset
    'num_workers': 4          # Number of workers for the DataLoader
}
