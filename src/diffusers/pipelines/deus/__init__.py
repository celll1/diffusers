# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from ...utils import _LazyModule


_import_structure = {
    "pipeline_deus": ["DeusPipeline"],
    "pipeline_deus_img2img": ["DeusImg2ImgPipeline"],
    "pipeline_deus_inpaint": ["DeusInpaintPipeline"],
    "pipeline_deus_multimodal": ["DeusMultiModalPipeline"],
    "pipeline_output": ["DeusPipelineOutput"],
}


if TYPE_CHECKING:
    from .pipeline_deus import DeusPipeline
    from .pipeline_deus_img2img import DeusImg2ImgPipeline
    from .pipeline_deus_inpaint import DeusInpaintPipeline
    from .pipeline_deus_multimodal import DeusMultiModalPipeline
    from .pipeline_output import DeusPipelineOutput
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
