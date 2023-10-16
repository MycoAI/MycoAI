import wandb
from .classification import ClassificationTask
from .pretraining import MLMTask, NSPTask

# Set up weights & biases
wandb.login('allow')