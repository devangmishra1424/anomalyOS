#!/usr/bin/env python3
"""
Check 1: Kaggle and DagHub Authentication

Verify that Kaggle API and DagHub credentials are properly configured.
"""
import os
import sys
from pathlib import Path

def check_kaggle_auth():
    """Check if Kaggle API is configured"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    print("Checking Kaggle authentication...")
    
    if kaggle_json.exists():
        print("✓ Kaggle credentials found")
        return True
    else:
        print("✗ Kaggle credentials not found")
        print(f"  Expected location: {kaggle_json}")
        print("  Download from: https://www.kaggle.com/settings/account")
        return False

def check_dagshub_auth():
    """Check if DagHub token is configured"""
    print("Checking DagHub authentication...")
    
    token = os.getenv('DAGSHUB_TOKEN')
    if token:
        print("✓ DagHub token found in environment")
        return True
    else:
        print("✗ DagHub token not found")
        print("  Set DAGSHUB_TOKEN environment variable")
        return False

def main():
    print("=" * 50)
    print("Authentication Check")
    print("=" * 50)
    
    kaggle_ok = check_kaggle_auth()
    dagshub_ok = check_dagshub_auth()
    
    print("\n" + "=" * 50)
    if kaggle_ok and dagshub_ok:
        print("✓ All authentication checks passed")
        return 0
    else:
        print("✗ Some authentication checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
