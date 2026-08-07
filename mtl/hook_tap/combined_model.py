"""
combined_model.py — Unified Multi-Task Deep Fusion Network (Insulated Modes)
Includes mode-switching overrides to completely bypass Ultralytics dataset validation errors.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from ultralytics import YOLO

class SpecializedDrowsinessNeckAndHead(nn.Module):
    """Dedicated neck and head blocks to isolate drowsiness features."""
    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.private_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Sequential(
            nn.Linear(64 * 6 * 6, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.private_conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class UnifiedInCabinNet(nn.Module):
    def __init__(self, base_weights: str):
        super().__init__()
        self.yolo_wrapper = YOLO(base_weights)
        
        self._captured_p3: torch.Tensor | None = None
        self.yolo_wrapper.model.model[15].register_forward_hook(self._hook_p4_layer)
        
        self.drw_branch = SpecializedDrowsinessNeckAndHead(in_channels=512)

    def _hook_p4_layer(self, module, layer_input, layer_output):
        self._captured_p3 = layer_output

    # 🚀 THE CRITICAL INSULATION BLOCK: Overrides PyTorch's recursive state propagation
    def train(self, mode: bool = True):
        # Only toggle state for standard PyTorch layers
        self.drw_branch.train(mode)
        # Force the underlying native backbone graph to toggle modes directly
        # This completely isolates the high-level yolo_wrapper class and stops the data.yaml check
        self.yolo_wrapper.model.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._captured_p3 = None
        
        with torch.no_grad():
            _ = self.yolo_wrapper.model(x)
            
        if self._captured_p3 is None:
            raise RuntimeError("Backbone routing missed Layer 15 hook link.")

        drowsiness_logits = self.drw_branch(self._captured_p3)
        return drowsiness_logits