"""
Utils package for Stock Prediction Application.

This package contains utility modules for parameter management,
configuration, training, evaluation, and web interface.
"""

from .parameters import (
    TrainingParameters,
    EvaluationParameters,
    WebParameters,
    parse_arguments,
    create_web_parameters,
    create_argument_parser,
    create_training_parameters,
    create_evaluation_parameters,
)

from .web import run_web_interface
from .training import run_training
from .evaluation import run_evaluation

__all__ = [
    # Parameter classes
    'TrainingParameters',
    'EvaluationParameters',
    'WebParameters',

    # Parameter functions
    'create_argument_parser',
    'parse_arguments',
    'create_training_parameters',
    'create_evaluation_parameters',
    'create_web_parameters',

    # Main execution functions
    'run_training',
    'run_evaluation',
    'run_web_interface'
]
