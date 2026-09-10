"""Training contracts independent of detection algorithms."""
from .connector import TrainingInput, training_input
from .trainer import TrainConfig, train_classifier

__all__ = ['TrainingInput', 'training_input', 'TrainConfig', 'train_classifier']
