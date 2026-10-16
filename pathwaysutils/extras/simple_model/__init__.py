# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simple Transformer model package using Flax NNX and Grain for Pathways."""

from pathwaysutils.extras.simple_model.compile import CompiledStepArtifacts
from pathwaysutils.extras.simple_model.compile import compile_training_step
from pathwaysutils.extras.simple_model.compile import compute_mesh_shape
from pathwaysutils.extras.simple_model.compile import update_step
from pathwaysutils.extras.simple_model.config import CONFIGS
from pathwaysutils.extras.simple_model.config import DEFAULT_CONFIG
from pathwaysutils.extras.simple_model.config import ModelConfig
from pathwaysutils.extras.simple_model.config import parse_topology
from pathwaysutils.extras.simple_model.data import create_data_iterator
from pathwaysutils.extras.simple_model.data import create_token_dataset
from pathwaysutils.extras.simple_model.data import SyntheticTokenSource
from pathwaysutils.extras.simple_model.model import geglu
from pathwaysutils.extras.simple_model.model import SimpleModel
from pathwaysutils.extras.simple_model.model import TransformerBlock
from pathwaysutils.extras.simple_model.workload import Workload

__all__ = [
    "CONFIGS",
    "CompiledStepArtifacts",
    "DEFAULT_CONFIG",
    "ModelConfig",
    "SimpleModel",
    "SyntheticTokenSource",
    "TransformerBlock",
    "Workload",
    "compile_training_step",
    "compute_mesh_shape",
    "create_data_iterator",
    "create_token_dataset",
    "geglu",
    "parse_topology",
    "update_step",
]
