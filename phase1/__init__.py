"""Phase 1: evaluation framework for the video-to-frame optimizer.

Determines whether the Phase 0/0.1 frame-selection strategy preserves the
information a downstream 3D reconstruction system needs, and by how much
frame count can be reduced before reconstruction quality degrades.

This package does not perform 3D reconstruction. It treats reconstruction as
an external, pluggable backend (see evaluation/reconstruction_backend.py).
"""
