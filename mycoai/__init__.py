import wandb
from mycoai import utils

# Set up device (GPU if available)
#utils.set_device('cuda')

# Set up weights & biases
wandb.login('allow')