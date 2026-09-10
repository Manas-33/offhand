"""Offhand M0c tool-calling eval harness.

Quantization-agnostic: it measures the tool-calling reliability of whatever
model a runner exposes. Swap the runner (FP16, quantized checkpoint, on-device
bridge) and the scoring stays identical.
"""
