#!/usr/bin/env python3
"""
Check 2: Hugging Face Spaces and Docker

Verify that Hugging Face Spaces environment and Docker are properly configured.
"""
import os
import sys
import subprocess

def check_hf_spaces_env():
    """Check if running in Hugging Face Spaces environment"""
    print("Checking Hugging Face Spaces environment...")
    
    hf_token = os.getenv('HF_API_TOKEN')
    hf_user = os.getenv('HF_USER')
    
    if hf_token:
        print("✓ HF_API_TOKEN found")
    else:
        print("✗ HF_API_TOKEN not found")
    
    if hf_user:
        print("✓ HF_USER found")
    else:
        print("✗ HF_USER not found")
    
    return bool(hf_token)

def check_docker():
    """Check if Docker is available"""
    print("Checking Docker installation...")
    
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Docker found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("✗ Docker not found")
    return False

def main():
    print("=" * 50)
    print("HF Spaces & Docker Check")
    print("=" * 50)
    
    hf_ok = check_hf_spaces_env()
    docker_ok = check_docker()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  HF Spaces: {'✓ OK' if hf_ok else '✗ MISSING'}")
    print(f"  Docker: {'✓ OK' if docker_ok else '✗ MISSING'}")
    
    return 0 if (hf_ok or docker_ok) else 1

if __name__ == "__main__":
    sys.exit(main())
