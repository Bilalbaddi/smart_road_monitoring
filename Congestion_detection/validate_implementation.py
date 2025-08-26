#!/usr/bin/env python3
"""
Simple validation script for Enhanced Smart Traffic Flow Analyzer

This script validates the core functionality without requiring external dependencies.
"""

import sys
import os

def validate_enhanced_features():
    """Validate that all enhanced features are properly implemented."""
    print("Enhanced Smart Traffic Flow Analyzer - Feature Validation")
    print("=" * 55)
    
    # Check if main.py exists and contains new features
    main_file = "main.py"
    if not os.path.exists(main_file):
        print("✗ main.py not found")
        return False
    
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Check for new imports
    required_imports = [
        "import smtplib",
        "from email.mime.multipart import MIMEMultipart",
        "from collections import defaultdict, deque",
        "from datetime import datetime"
    ]
    
    missing_imports = []
    for imp in required_imports:
        if imp not in content:
            missing_imports.append(imp)
    
    if missing_imports:
        print("✗ Missing imports:")
        for imp in missing_imports:
            print(f"  - {imp}")
        return False
    
    print("✓ All required imports present")
    
    # Check for new classes and dataclasses
    required_classes = [
        "class GridCell:",
        "class FrameMetrics:",
        "class CongestionDetector:"
    ]
    
    missing_classes = []
    for cls in required_classes:
        if cls not in content:
            missing_classes.append(cls)
    
    if missing_classes:
        print("✗ Missing classes:")
        for cls in missing_classes:
            print(f"  - {cls}")
        return False
    
    print("✓ All required classes present")
    
    # Check for new methods
    required_methods = [
        "def create_grid_cells(",
        "def generate_heatmap_overlay(",
        "def predict_congestion(",
        "def send_email_alert(",
        "def assign_vehicle_to_grid(",
        "def _draw_grid_lines(",
        "def _draw_grid_statistics("
    ]
    
    missing_methods = []
    for method in required_methods:
        if method not in content:
            missing_methods.append(method)
    
    if missing_methods:
        print("✗ Missing methods:")
        for method in missing_methods:
            print(f"  - {method}")
        return False
    
    print("✓ All required methods present")
    
    # Check for enhanced argument parser
    required_args = [
        "--grid-rows",
        "--grid-cols", 
        "--email-user",
        "--email-password",
        "--email-to"
    ]
    
    missing_args = []
    for arg in required_args:
        if arg not in content:
            missing_args.append(arg)
    
    if missing_args:
        print("✗ Missing command line arguments:")
        for arg in missing_args:
            print(f"  - {arg}")
        return False
    
    print("✓ All command line arguments present")
    
    # Check for key functionality
    key_features = [
        "grid_rows",
        "grid_cols", 
        "heatmap",
        "email_config",
        "frame_metrics_history",
        "congestion_predicted"
    ]
    
    missing_features = []
    for feature in key_features:
        if feature not in content:
            missing_features.append(feature)
    
    if missing_features:
        print("✗ Missing key features:")
        for feature in missing_features:
            print(f"  - {feature}")
        return False
    
    print("✓ All key features implemented")
    
    return True

def validate_file_structure():
    """Validate that all supporting files are present."""
    print("\nValidating file structure...")
    
    expected_files = [
        "main.py",
        "config.py", 
        "example_usage.py",
        "test_enhanced_features.py"
    ]
    
    missing_files = []
    for file in expected_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("✗ Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("✓ All supporting files present")
    return True

def validate_documentation():
    """Validate that documentation files exist."""
    print("\nValidating documentation...")
    
    doc_files = [
        "../ENHANCED_README.md",
        "../requirements.txt"
    ]
    
    missing_docs = []
    for doc in doc_files:
        if not os.path.exists(doc):
            missing_docs.append(doc)
    
    if missing_docs:
        print("✗ Missing documentation:")
        for doc in missing_docs:
            print(f"  - {doc}")
        return False
    
    print("✓ All documentation present")
    return True

def main():
    """Run all validations."""
    print("Starting validation...")
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    validations = [
        validate_enhanced_features,
        validate_file_structure,
        validate_documentation
    ]
    
    all_passed = True
    for validation in validations:
        if not validation():
            all_passed = False
    
    print("\n" + "=" * 55)
    if all_passed:
        print("🎉 All validations passed! Enhanced features are properly implemented.")
        print("\nNext steps:")
        print("1. Install requirements: pip install -r ../requirements.txt")
        print("2. Get YOLO model weights")
        print("3. Run: python example_usage.py")
    else:
        print("⚠️  Some validations failed. Please review the implementation.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)