# -*- coding: utf-8 -*-

"""
Preprocessing module for generating / preparing inputs for ion dynamics simulations with IDSimF

Modules:
  * ion_cloud_generation: Generation of ion cloud initialization files
  * field_generation: Generation / Transformation of scalar and vector fields
"""

from .comsol_import import *
from .field_generation import *
from .ion_cloud_generation import *
from .run_configuration_preprocessing import *


