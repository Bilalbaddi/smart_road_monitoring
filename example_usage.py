#!/usr/bin/env python3
"""
Enhanced Smart Traffic Flow Analyzer - Example Usage Script

This script demonstrates how to use the enhanced traffic analysis system with
grid-based congestion detection, heatmap visualization, prediction, and email alerts.
"""

import subprocess
import sys
import os

def run_basic_analysis():
    """Run basic traffic analysis with default grid settings."""
    print("Running basic traffic analysis with 4x4 grid...")
    
    cmd = [
        sys.executable, "main.py",
        "--weights", "../Model yolo11x/yolo11x.pt",  # Update with your model path
        "--input", "traffic_video.mp4",
        "--output", "output_basic.mp4",
        "--grid-rows", "4",
        "--grid-cols", "4"
    ]
    
    subprocess.run(cmd)
    print("Basic analysis complete. Output: output_basic.mp4")

def run_detailed_analysis():
    """Run detailed traffic analysis with higher resolution grid."""
    print("Running detailed traffic analysis with 8x8 grid...")
    
    cmd = [
        sys.executable, "main.py",
        "--weights", "../Model yolo11x/yolo11x.pt",  # Update with your model path
        "--input", "traffic_video.mp4",
        "--output", "output_detailed.mp4",
        "--grid-rows", "8",
        "--grid-cols", "8"
    ]
    
    subprocess.run(cmd)
    print("Detailed analysis complete. Output: output_detailed.mp4")

def run_with_email_alerts():
    """Run analysis with email notification system enabled."""
    print("Running analysis with email alerts...")
    
    # Note: Replace these with your actual email credentials
    email_user = input("Enter your email address: ")
    email_password = input("Enter your email password (or app-specific password): ")
    email_to = input("Enter recipient email address: ")
    
    cmd = [
        sys.executable, "main.py",
        "--weights", "../Model yolo11x/yolo11x.pt",  # Update with your model path
        "--input", "traffic_video.mp4",
        "--output", "output_with_alerts.mp4",
        "--grid-rows", "6",
        "--grid-cols", "6",
        "--email-user", email_user,
        "--email-password", email_password,
        "--email-to", email_to,
        "--email-host", "smtp.gmail.com",
        "--email-port", "587"
    ]
    
    subprocess.run(cmd)
    print("Analysis with email alerts complete. Output: output_with_alerts.mp4")

def run_custom_configuration():
    """Run analysis with custom SMTP settings."""
    print("Running analysis with custom configuration...")
    
    cmd = [
        sys.executable, "main.py",
        "--weights", "../Model yolo11x/yolo11x.pt",  # Update with your model path
        "--input", "traffic_video.mp4",
        "--output", "output_custom.mp4",
        "--grid-rows", "5",
        "--grid-cols", "5",
        "--email-host", "smtp.outlook.com",  # For Outlook/Hotmail
        "--email-port", "587"
    ]
    
    subprocess.run(cmd)
    print("Custom configuration analysis complete. Output: output_custom.mp4")

def main():
    """Main function to demonstrate different usage scenarios."""
    print("Enhanced Smart Traffic Flow Analyzer - Example Usage")
    print("=" * 55)
    
    print("\nAvailable examples:")
    print("1. Basic analysis (4x4 grid, no email)")
    print("2. Detailed analysis (8x8 grid, no email)")
    print("3. Analysis with email alerts")
    print("4. Custom configuration")
    print("5. Exit")
    
    while True:
        choice = input("\nSelect an option (1-5): ").strip()
        
        if choice == "1":
            run_basic_analysis()
        elif choice == "2":
            run_detailed_analysis()
        elif choice == "3":
            run_with_email_alerts()
        elif choice == "4":
            run_custom_configuration()
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select 1-5.")

if __name__ == "__main__":
    # Ensure we're in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check if required files exist
    if not os.path.exists("main.py"):
        print("Error: main.py not found in current directory")
        sys.exit(1)
    
    if not os.path.exists("traffic_video.mp4"):
        print("Warning: traffic_video.mp4 not found. Please ensure you have a video file.")
    
    main()