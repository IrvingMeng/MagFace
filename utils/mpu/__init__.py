#!/usr/bin/env python
# Copyright (c) Aibee, Inc. and its affiliates. All Rights Reserved

"""
Model parallel utility interfaces.
"""

from .losses import ParallelCrossEntropyLoss
from .losses import ParallelArcMarginLoss

from .grads import clip_grad_norm

from .initialize import model_parallel_is_initialized
from .initialize import destroy_model_parallel
from .initialize import initialize_model_parallel

from .initialize import get_model_parallel_group
from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_src_rank
from .initialize import get_model_parallel_world_size

from .initialize import get_parameter_parallel_group
from .initialize import get_parameter_parallel_rank
from .initialize import get_parameter_parallel_world_size

from .layers import ParallelLinear
from .layers import ParallelArcMarginLinear
from .layers import accuracy
from .layers import batch_shuffle
from .layers import batch_unshuffle

from .mappings import scatter_to_region
from .mappings import copy_to_region
from .mappings import gather_from_region
from .mappings import reduce_from_region
from .mappings import _gather
from .mappings import _reduce
from .mappings import _split

from .random import model_parallel_cuda_manual_seed

from .ext_layers import ParallelMagLinear
from .ext_layers import ParallelCosMarginLinear
from .ext_losses import ParallelMagLoss
from .ext_losses import ParallelCosMarginLoss