#!/usr/bin/env python3
"""
Deployment script for Stock Prediction Application.

This script helps deploy the application to various platforms.
"""


import sys
import subprocess
from pathlib import Path


def check_requirements():
    """Check if all required files exist for deployment."""
    required_files = [
        "app.py",
        "pyproject.toml",
        "uv.lock",
        ".streamlit/config.toml",
        "Dockerfile",
        "render.yaml"
    ]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print("❌ Missing required files for deployment:")
        for file in missing_files:
            print(f"   - {file}")
        return False

    print("✅ All required files present for deployment")
    return True


def test_local():
    """Test the application locally before deployment."""
    print("🧪 Testing application locally...")

    try:
        # Test imports
        result = subprocess.run([
            sys.executable, "-c",
            "import app; print('✅ App imports successfully')"
        ], capture_output=True, text=True, cwd=Path.cwd())

        if result.returncode == 0:
            print("✅ Application imports successfully")
            return True
        else:
            print(f"❌ Import test failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Local test failed: {e}")
        return False


def main():
    """Main deployment preparation function."""
    print("🚀 Stock Prediction App - Deployment Preparation")
    print("=" * 50)

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Test locally
    if not test_local():
        print("❌ Local tests failed. Please fix issues before deploying.")
        sys.exit(1)

    print("\n✅ Deployment preparation complete!")
    print("\n📋 Next steps for Render deployment:")
    print("1. Push your code to GitHub")
    print("2. Connect your GitHub repo to Render")
    print("3. Use the following settings in Render:")
    print("   - Build Command: pip install uv && uv sync")
    print("   - Start Command: uv run streamlit run app.py --server.address 0.0.0.0 --server.port $PORT")
    print("   - Environment: Python 3.11")
    print("\n🐳 For Docker deployment:")
    print("   - Use the provided Dockerfile")
    print("   - Ensure port 8501 is exposed")


if __name__ == "__main__":
    main()
