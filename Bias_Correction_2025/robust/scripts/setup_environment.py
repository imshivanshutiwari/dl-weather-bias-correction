#!/usr/bin/env python
"""
Environment Setup and Dependency Checker
==========================================

Run this script FIRST on your lab system to:
1. Check Python version
2. Check all required libraries
3. Install missing dependencies automatically
4. Verify GPU/CUDA availability
5. Run quick validation tests

Usage:
    python setup_environment.py

Author: IITM Pune - Government of India Project
"""

import subprocess
import sys
import os
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# Required packages with minimum versions
REQUIRED_PACKAGES = {
    # Core ML
    'torch': '2.0.0',
    'numpy': '1.21.0',
    
    # Data loading
    'xarray': '2023.1.0',
    'netCDF4': '1.6.0',
    'pandas': '1.5.0',
    
    # Visualization
    'matplotlib': '3.5.0',
    
    # Configuration
    'pyyaml': '6.0',
    
    # Utilities
    'scipy': '1.9.0',
    'tensorboard': '2.10.0',
    
    # GNN (PyTorch Geometric) - CRITICAL FOR NOVELTY!
    'torch_geometric': '2.3.0',
}

# Optional packages (nice to have)
OPTIONAL_PACKAGES = {
    'cartopy': '0.21.0',  # For map projections
    'geopandas': '0.12.0',  # For geographic data
    'wandb': '0.13.0',  # For experiment tracking
    'tqdm': '4.64.0',  # Progress bars
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_status(name, status, extra=""):
    if status == "OK":
        icon = ""
    elif status == "MISSING":
        icon = ""
    elif status == "WARNING":
        icon = ""
    else:
        icon = ""
    
    if extra:
        print(f"  {icon} {name}: {status} ({extra})")
    else:
        print(f"  {icon} {name}: {status}")


def check_python_version():
    """Check Python version."""
    print_header("Checking Python Version")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 8:
        print_status("Python", "OK", version_str)
        return True
    else:
        print_status("Python", "WARNING", f"{version_str} - Recommend 3.8+")
        return True


def get_installed_version(package_name):
    """Get installed version of a package."""
    try:
        # Handle special package name mappings
        import_name_map = {
            'pyyaml': 'yaml',
            'torch_geometric': 'torch_geometric',
            'netCDF4': 'netCDF4',
        }
        
        import_name = import_name_map.get(package_name, package_name)
        
        # Try importing
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return version
    except ImportError:
        return None


def install_package(package_name, version=None):
    """Install a package using pip."""
    if package_name == 'torch':
        # Special handling for PyTorch
        print(f"    Installing PyTorch (this may take a while)...")
        cmd = [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio']
    elif package_name == 'torch_geometric':
        # Special handling for PyG
        print(f"    Installing PyTorch Geometric (this may take a while)...")
        cmd = [sys.executable, '-m', 'pip', 'install', 'torch_geometric']
    else:
        pkg = f"{package_name}>={version}" if version else package_name
        cmd = [sys.executable, '-m', 'pip', 'install', pkg]
    
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        return False


def check_and_install_packages(packages, optional=False):
    """Check packages and install missing ones."""
    
    missing = []
    installed = []
    
    for package, min_version in packages.items():
        version = get_installed_version(package)
        
        if version is None:
            missing.append((package, min_version))
            print_status(package, "MISSING", f"need >={min_version}")
        else:
            installed.append((package, version))
            print_status(package, "OK", version)
    
    if missing:
        if optional:
            print(f"\n  Optional packages missing: {len(missing)}")
            response = input("  Install optional packages? [y/N]: ").strip().lower()
            if response != 'y':
                return True
        
        print(f"\n  Installing {len(missing)} missing packages...")
        
        for package, version in missing:
            print(f"    Installing {package}...")
            success = install_package(package, version)
            
            if success:
                print_status(f"    {package}", "INSTALLED")
            else:
                print_status(f"    {package}", "FAILED")
                if not optional:
                    return False
    
    return True


def check_gpu():
    """Check GPU and CUDA availability."""
    print_header("Checking GPU/CUDA")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            
            print_status("CUDA", "OK", cuda_version)
            print_status("GPU", "OK", f"{gpu_count}x {gpu_name}")
            
            # Check memory
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print_status(f"  GPU {i} Memory", "OK", f"{memory_gb:.1f} GB")
            
            return True
        else:
            print_status("CUDA", "WARNING", "Not available - will use CPU")
            print("   Training will be slower without GPU!")
            return True
            
    except ImportError:
        print_status("PyTorch", "MISSING", "Install PyTorch first")
        return False


def check_torch_geometric():
    """Check if PyTorch Geometric is properly installed."""
    print_header("Checking PyTorch Geometric (for GNN)")
    
    try:
        import torch_geometric
        from torch_geometric.nn import GATConv
        
        print_status("torch_geometric", "OK", torch_geometric.__version__)
        print_status("GATConv", "OK", "GNN layers available")
        return True
        
    except ImportError as e:
        print_status("torch_geometric", "MISSING", str(e))
        print("\n  Installing PyTorch Geometric...")
        
        try:
            # Install PyG
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 'torch_geometric'
            ])
            
            try:
                # Install optional dependencies for better performance
                # Note: These are very sensitive to version mismatches, so we make them optional
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', 
                    'pyg_lib', 'torch_scatter', 'torch_sparse', 'torch_cluster', 'torch_spline_conv',
                    '-f', 'https://data.pyg.org/whl/torch-2.4.0+cu121.html'
                ])
                print_status("PyG Extras", "INSTALLED")
            except subprocess.CalledProcessError:
                print_status("PyG Extras", "SKIPPED", "Optional dependencies failed (OK)")
            
            print_status("torch_geometric", "INSTALLED")
            return True
            
        except subprocess.CalledProcessError:
            print_status("torch_geometric", "FAILED", "Manual install may be needed")
            print("\n  Try manually: pip install torch_geometric")
            return False


def run_quick_test():
    """Run quick test to verify everything works."""
    print_header("Running Quick Verification Test")
    
    try:
        # Add project to path (go up from scripts -> robust -> Bias_Correction_2025)
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        print("  Testing imports...")
        
        # Test core imports
        from robust.models.msua_net import MSUANet
        print_status("MSUANet import", "OK")
        
        from robust.losses.compound import CompoundBiasCorrectionLoss
        print_status("CompoundLoss import", "OK")
        
        from robust.data.datasets import RobustNormalizer
        print_status("Dataset import", "OK")
        
        from robust.training.trainer import BiasCorrectionTrainer
        print_status("Trainer import", "OK")
        
        # Try GNN import
        try:
            from robust.models.gnn import SpatialGraphBlock
            print_status("GNN import", "OK")
        except ImportError:
            print_status("GNN import", "WARNING", "GNN features disabled")
        
        # Test model creation
        print("\n  Testing model creation...")
        import torch
        model = MSUANet(in_channels=10, base_filters=32, num_levels=2, uncertainty=False)
        model.eval()  # Set to eval mode to avoid BatchNorm issues
        x = torch.randn(2, 10, 64, 64)  # Larger batch and spatial size
        with torch.no_grad():
            out, _ = model(x)
        print_status("Model forward pass", "OK", f"output shape: {out.shape}")
        
        print("\n   All tests passed!")
        return True
        
    except Exception as e:
        print_status("Quick test", "FAILED", str(e))
        return False


def create_requirements_file():
    """Create requirements.txt file."""
    print_header("Creating requirements.txt")
    
    requirements = []
    
    for package, version in REQUIRED_PACKAGES.items():
        if package == 'torch':
            requirements.append("# Install PyTorch separately: https://pytorch.org")
            requirements.append("# torch>=2.0.0")
        elif package == 'torch_geometric':
            requirements.append("# Install PyG separately: https://pytorch-geometric.readthedocs.io")
            requirements.append("# torch_geometric>=2.3.0")
        else:
            requirements.append(f"{package}>={version}")
    
    requirements_path = Path(__file__).parent / 'requirements.txt'
    with open(requirements_path, 'w') as f:
        f.write('\n'.join(requirements))
    
    print(f"  Created: {requirements_path}")


def main():
    print("\n" + "=" * 60)
    print("   IITM Pune - Government of India")
    print("  Environment Setup for Bias Correction System")
    print("=" * 60)
    
    all_ok = True
    
    # 1. Check Python
    if not check_python_version():
        all_ok = False
    
    # 2. Check and install required packages
    print_header("Checking Required Packages")
    if not check_and_install_packages(REQUIRED_PACKAGES, optional=False):
        all_ok = False
    
    # 3. Check GPU
    check_gpu()
    
    # 4. Check PyTorch Geometric (for GNN)
    check_torch_geometric()
    
    # 5. Create requirements.txt
    create_requirements_file()
    
    # 6. Run quick test
    if all_ok:
        if not run_quick_test():
            all_ok = False
    
    # Summary
    print_header("Setup Summary")
    
    if all_ok:
        print("""
   Environment is ready!
  
  Next steps:
  1. Update data paths in: robust/config/production.yaml
  2. Run tests: python robust/scripts/run_all_tests.py
  3. Start training: python robust/scripts/train.py --config robust/config/production.yaml
  
   Good luck with your training!
""")
    else:
        print("""
   Some issues were found.
  
  Please fix the issues above and run this script again.
  
  For manual installation:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install torch_geometric
    pip install -r requirements.txt
""")
    
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
