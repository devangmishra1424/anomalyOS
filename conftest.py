# conftest.py
import sys
import os

# Add project root to Python path so imports work in CI
sys.path.insert(0, os.path.dirname(__file__))