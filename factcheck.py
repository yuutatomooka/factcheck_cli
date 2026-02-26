#!/usr/bin/env python3
"""Launcher: run factchecker.factcheck.main(). Use: python factcheck.py or python factchecker/factcheck.py"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factchecker.factcheck import main
sys.exit(main())
