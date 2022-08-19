# -*- coding: utf-8 -*-

"""
Preprocessing module for generating / preparing inputs for ion dynamics simulations with IDSimF

Modules:
  * ion_cloud_generation: Generation of ion cloud initialization files
  * field_generation: Generation / Transformation of scalar and vector fields
"""

from . import comsol_import
from . import field_generation
from . import input_file_preprocessing
from .input_file_preprocessing import generate_input_files_from_template

from . import ion_cloud_generation
