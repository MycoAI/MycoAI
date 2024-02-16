import wandb
from mycoai import utils

# Set up device (GPU if available)
utils.set_device('cuda')

# Set up weights & biases
wandb.login('allow', key='07c4f4b038b577ca5aa830979d6943aecd9ebec4')