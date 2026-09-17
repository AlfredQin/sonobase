"""US-RFDETR — per-dataset detector training & evaluation.

Two recipes:

* :class:`TrainUSRFDETRRecipe` (in :mod:`.train`) — DDP-aware training
  with per-epoch validation and best-checkpoint tracking.
* :class:`TestUSRFDETRRecipe` (in :mod:`.test`) — single-GPU evaluation
  loading a `best.pt` produced by training.

Each recipe is invoked via the corresponding `main()` (Hydra entry point).
"""
