# Configuration file for Smart Traffic Flow Analyzer
# This file contains default configuration parameters for the enhanced traffic analysis system

# Grid Configuration
DEFAULT_GRID_ROWS = 4
DEFAULT_GRID_COLS = 4

# Congestion Detection Thresholds
VEHICLE_COUNT_THRESHOLD = 6  # Minimum vehicles to consider congestion
SPEED_THRESHOLD = 75  # Maximum average speed (px/s) to consider congestion
SPEED_WINDOW = 30  # Number of frames to track speed history

# Prediction Parameters
FRAME_HISTORY_SIZE = 10  # Number of frames to keep for trend analysis
SPEED_DECREASE_THRESHOLD = -5  # Minimum speed decrease (px/s per frame) for prediction
DENSITY_INCREASE_THRESHOLD = 0.5  # Minimum density increase (vehicles per frame) for prediction

# Email Configuration
EMAIL_COOLDOWN = 300  # Seconds between email alerts (5 minutes)
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

# Visualization Parameters
HEATMAP_ALPHA = 0.4  # Transparency level for heatmap overlay (0.0 - 1.0)
GRID_LINE_COLOR = (255, 255, 255)  # White color for grid lines
GRID_LINE_THICKNESS = 1

# Color Mapping for Heatmap
# Colors are in BGR format (Blue, Green, Red)
LOW_DENSITY_COLOR = (0, 255, 0)      # Green
MEDIUM_DENSITY_COLOR = (0, 255, 255) # Yellow
HIGH_DENSITY_COLOR = (0, 0, 255)     # Red

# Density Thresholds for Color Mapping
LOW_DENSITY_THRESHOLD = 0.3   # Below this ratio = green
MEDIUM_DENSITY_THRESHOLD = 0.7 # Below this ratio = yellow/orange, above = red

# Vehicle Classes (YOLO classes for vehicles)
VEHICLE_CLASSES = {
    0: "person",
    1: "bicycle", 
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck"
}

# Class-specific colors for bounding boxes (BGR format)
CLASS_COLORS = {
    0: (0, 255, 0),    # Green for person
    1: (255, 0, 0),    # Blue for bicycle
    2: (0, 255, 255),  # Yellow for car
    3: (255, 255, 0),  # Cyan for motorcycle
    4: (255, 0, 255),  # Magenta for airplane
    5: (0, 100, 100),  # Dark cyan for bus
    6: (100, 0, 100),  # Dark magenta for train
    7: (100, 100, 0)   # Dark yellow for truck
}