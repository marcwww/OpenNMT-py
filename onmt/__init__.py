""" Main entry point of the ONMT library """
from __future__ import division, print_function

import onmt.encoders
import onmt.models
import onmt.utils
import onmt.modules
from onmt.trainer import Trainer

# For Flake
__all__ = [onmt.encoders, onmt.models,
           onmt.utils, onmt.modules, "Trainer"]

__version__ = "0.4.0"
