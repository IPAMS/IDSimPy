# -*- coding: utf-8 -*-

"""
Preprocessing module for generating / preparing inputs for ion dynamics simulations with IDSimF

Modules:
  * ion_cloud_generation: Generation of ion cloud initialization files
  * field_generation: Generation / Transformation of scalar and vector fields
"""

from . import comsol_import
from . import field_generation
from . import run_configuration_preprocessing
from .run_configuration_preprocessing import generate_run_configurations_from_template

from . import ion_cloud_generation
