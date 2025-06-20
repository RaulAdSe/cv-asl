#!/usr/bin/env python3
"""
Live ASL Recognition - Main Runner Script

Quick launcher for the live ASL recognition system.
This script integrates the camera hand detection with the trained model.

Requirements:
- Trained model must exist: src/asl_dl/models/asl_abc_model.pth
- Camera connected and accessible
- All dependencies installed (opencv-python, torch, torchvision)

Usage:
    python run_live_asl.py

Controls in live mode:
- Q: Quit
- S: Toggle statistics display
- R: Reset hand tracker
- SPACE: Pause/unpause
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and run
from asl_cam.live_asl import main

if __name__ == "__main__":
    print("🚀 Launching Live ASL Recognition System...")
    print("📋 Controls: Q=Quit | S=Stats | R=Reset | SPACE=Pause")
    print("=" * 50)
    main() 