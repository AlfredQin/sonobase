"""US-RFDETR model components.

Public surface:

- :func:`build_rfdetr` — top-level factory: builds the SAM2 backbone
  adapter + RF-DETR transformer + heads + criterion + postprocessor and
  returns a dict ``{"model", "criterion", "postprocess"}``.
- :class:`SAM2RFDETRBackbone` — wraps a SAM2 ``ImageEncoder`` so its
  output matches RF-DETR's ``forward(NestedTensor) -> List[NestedTensor]``
  contract.
- :class:`SAM2Joiner` — pairs the backbone with a position-encoding module.
- :func:`build_sam2_rfdetr_backbone` — convenience constructor for the
  Joiner.

The full RF-DETR model implementation (LWDETR + transformer + matcher +
ops + segmentation head + utilities) lives next to this module —
:mod:`.lwdetr`, :mod:`.transformer`, :mod:`.matcher`, :mod:`.ops`,
:mod:`.segmentation_head`, :mod:`.position_encoding`, :mod:`.util` —
ported near-verbatim from the upstream Roboflow / LW-DETR codebase,
with import paths rewritten to the sonobase namespace.
"""

from nemo_cv.components.models.rfdetr.builder import build_rfdetr
from nemo_cv.components.models.rfdetr.sam2_backbone import (
    SAM2Joiner,
    SAM2RFDETRBackbone,
    build_sam2_rfdetr_backbone,
)

__all__ = [
    "build_rfdetr",
    "SAM2RFDETRBackbone",
    "SAM2Joiner",
    "build_sam2_rfdetr_backbone",
]
