#!/usr/bin/env python3
"""
Run Warp AV — simple entry point.
Usage: python3 run.py
"""
import sys
import os

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from warp_av.main import main

if __name__ == "__main__":
    main()
