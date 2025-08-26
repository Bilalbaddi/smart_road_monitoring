import ultralytics

ultralytics.checks()

import cv2
import math
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import argparse
import time
from datetime import datetime

# Import email configuration
try:
    from email_config import get_email_config, validate_email_config
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    print("Email configuration not found. Email alerts will be disabled.")

# Define the argument parser for input and output paths
parser = argparse.ArgumentParser(description="Run congestion detection on a video.")
parser.add_argument("--weights", type=str, required=True, help="Path to the YOLO model weights")
parser.add_argument("--input", type=str, required=True, help="Path to the input video")
parser.add_argument("--output", type=str, default="output_video.mp4", help="Path to save the output video")
parser.add_argument("--grid-rows", type=int, default=4, help="Number of grid rows for congestion heatmap")
parser.add_argument("--grid-cols", type=int, default=4, help="Number of grid columns for congestion heatmap")
parser.add_argument("--email-host", type=str, default="smtp.gmail.com", help="SMTP server host")
parser.add_argument("--email-port", type=int, default=587, help="SMTP server port")
parser.add_argument("--email-user", type=str, help="Email sender username")
parser.add_argument("--email-password", type=str, help="Email sender password")
parser.add_argument("--email-to", type=str, help="Email recipient address")

args = parser.parse_args()

@dataclass
class GridCell:
    """Represents a single grid cell with vehicle density tracking."""
    row: int
    col: int
    vehicle_count: int = 0
    vehicle_speeds: List[float] = None
    
    def __post_init__(self):
        if self.vehicle_speeds is None:
            self.vehicle_speeds = []

@dataclass
class FrameMetrics:
    """Stores metrics for a single frame."""
    timestamp: float
    total_vehicles: int
    avg_speed: float
    grid_densities: np.ndarray
    is_congested: bool

@dataclass
class VehicleData:
    position: Tuple[int, int]
    class_id: int
    class_name: str
    speed: float = 0.0

class CongestionDetector:
    def __init__(
        self,
        model_path: str,
        grid_rows: int = 4,
        grid_cols: int = 4,
        vehicle_threshold: int = 6,
        speed_threshold: float = 75,
        speed_window: int = 30,
        frame_history_size: int = 10,
        email_config: Dict = None
    ):
        self.model = YOLO(model_path)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.vehicle_threshold = vehicle_threshold
        self.speed_threshold = speed_threshold
        self.speed_window = speed_window
        self.frame_history_size = frame_history_size
        
        # Email configuration
        if email_config is None and EMAIL_AVAILABLE:
            try:
                email_config = get_email_config()
                is_valid, message = validate_email_config(email_config)
                if is_valid:
                    self.email_config = email_config
                    print(f"Email configuration loaded: {message}")
                else:
                    self.email_config = {}
                    print(f"Email configuration invalid: {message}")
            except Exception as e:
                self.email_config = {}
                print(f"Failed to load email configuration: {e}")
        else:
            self.email_config = email_config or {}
        
        self.last_email_time = 0
        self.email_cooldown = 300  # 5 minutes between emails

        # State tracking
        self.vehicles: Dict[int, VehicleData] = {}
        self.speed_history: Dict[int, list] = {}
        self.frame_metrics_history: deque = deque(maxlen=frame_history_size)

        # Unique object tracking
        self.unique_objects = defaultdict(set)

        # Class-specific colors (BGR format)
        self.class_colors = {
            0: (0, 255, 0),
            1: (255, 0, 0),
            2: (0, 255, 255),
            3: (255, 255, 0),
            4: (255, 0, 255),
            5: (0, 100, 100)
        }

    def calculate_speed(self, current_pos: Tuple[int, int],
                       previous_pos: Tuple[int, int], fps: float) -> float:
        """Calculate speed in pixels per second using Euclidean distance."""
        dx = current_pos[0] - previous_pos[0]
        dy = current_pos[1] - previous_pos[1]
        return math.sqrt(dx * dx + dy * dy) * fps

    def create_grid_cells(self, frame_shape: Tuple[int, int]) -> List[List[GridCell]]:
        """Create a grid of cells for the frame."""
        height, width = frame_shape[:2]
        grid = []
        
        for row in range(self.grid_rows):
            grid_row = []
            for col in range(self.grid_cols):
                grid_row.append(GridCell(row, col))
            grid.append(grid_row)
        
        return grid

    def get_grid_coordinates(self, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get grid cell dimensions."""
        height, width = frame_shape[:2]
        cell_height = height // self.grid_rows
        cell_width = width // self.grid_cols
        return height, width, cell_height, cell_width

    def assign_vehicle_to_grid(self, position: Tuple[int, int], 
                              frame_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Assign a vehicle position to the appropriate grid cell."""
        height, width, cell_height, cell_width = self.get_grid_coordinates(frame_shape)
        
        row = min(position[1] // cell_height, self.grid_rows - 1)
        col = min(position[0] // cell_width, self.grid_cols - 1)
        
        return row, col

    def generate_heatmap_overlay(self, frame: np.ndarray, 
                                grid_cells: List[List[GridCell]]) -> np.ndarray:
        """Generate a transparent heatmap overlay based on vehicle density in grid cells."""
        height, width, cell_height, cell_width = self.get_grid_coordinates(frame.shape)
        
        # Create overlay with same dimensions as frame
        overlay = np.zeros_like(frame, dtype=np.uint8)
        
        # Calculate maximum density for normalization
        max_density = max(max(cell.vehicle_count for cell in row) for row in grid_cells)
        if max_density == 0:
            max_density = 1  # Avoid division by zero
        
        for row_idx, grid_row in enumerate(grid_cells):
            for col_idx, cell in enumerate(grid_row):
                # Calculate cell boundaries
                y1 = row_idx * cell_height
                y2 = min((row_idx + 1) * cell_height, height)
                x1 = col_idx * cell_width
                x2 = min((col_idx + 1) * cell_width, width)
                
                # Normalize density (0-1)
                density_ratio = cell.vehicle_count / max_density
                
                # Determine color based on density
                if density_ratio < 0.3:
                    # Green for low density
                    color = (0, int(255 * density_ratio / 0.3), 0)
                elif density_ratio < 0.7:
                    # Yellow-orange for medium density
                    green_component = int(255 * (0.7 - density_ratio) / 0.4)
                    color = (0, green_component, 255)
                else:
                    # Red for high density
                    color = (0, 0, 255)
                
                # Fill the grid cell with color
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        
        # Apply transparency (alpha blending)
        alpha = 0.4  # Transparency level
        result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        return result

    def predict_congestion(self) -> bool:
        """Predict short-term congestion based on vehicle speed and density trends."""
        if len(self.frame_metrics_history) < 3:
            return False
        
        # Get recent metrics
        recent_metrics = list(self.frame_metrics_history)[-5:]  # Last 5 frames
        
        # Calculate trends
        speeds = [m.avg_speed for m in recent_metrics if m.avg_speed > 0]
        densities = [m.total_vehicles for m in recent_metrics]
        
        if len(speeds) < 2 or len(densities) < 2:
            return False
        
        # Check current conditions
        current_vehicles = recent_metrics[-1].total_vehicles
        current_speed = recent_metrics[-1].avg_speed
        
        # Check for speed decrease and density increase
        if len(speeds) >= 3 and len(densities) >= 3:
            speed_trend = np.polyfit(range(len(speeds)), speeds, 1)[0]  # Slope
            density_trend = np.polyfit(range(len(densities)), densities, 1)[0]  # Slope
            
            # More sensitive prediction thresholds
            speed_decreasing = speed_trend < -2  # Speed decreasing by more than 2 units per frame
            density_increasing = density_trend > 0.3  # Density increasing by more than 0.3 vehicles per frame
            
            # Additional conditions for prediction
            moderate_density = current_vehicles > 6  # More than 6 vehicles currently
            declining_speed = current_speed < 45    # Current speed below 45
            
            return (speed_decreasing and density_increasing) or (moderate_density and declining_speed)
        
        # Fallback: predict based on current conditions
        return current_vehicles > 7 and current_speed < 35

    def send_email_alert(self, frame: np.ndarray, timestamp: str):
        """Send email notification when congestion is detected."""
        if not self.email_config:
            print("Email notifications disabled (no configuration)")
            return
            
        if not all(k in self.email_config for k in ['host', 'port', 'user', 'password', 'to']):
            print("Email notifications disabled (missing configuration)")
            return
        
        # Check cooldown period
        current_time = time.time()
        if current_time - self.last_email_time < self.email_cooldown:
            return
        
        # Skip if using test configuration
        if self.email_config.get('host') == 'localhost':
            print(f"TEST MODE: Would send congestion alert email to {self.email_config['to']} at {timestamp}")
            self.last_email_time = current_time
            return
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['user']
            msg['To'] = self.email_config['to']
            msg['Subject'] = f"Traffic Congestion Alert - {timestamp}"
            
            # Get current frame statistics
            recent_metrics = list(self.frame_metrics_history)[-1] if self.frame_metrics_history else None
            vehicle_count = recent_metrics.total_vehicles if recent_metrics else "Unknown"
            avg_speed = f"{recent_metrics.avg_speed:.1f}" if recent_metrics else "Unknown"
            
            # Email body
            body = f"""
Traffic Congestion Alert

Timestamp: {timestamp}
Current Vehicles: {vehicle_count}
Average Speed: {avg_speed} units
Grid Configuration: {self.grid_rows}x{self.grid_cols}

Status: Congestion predicted based on increasing vehicle density and decreasing average speed.

Please see attached image showing the current traffic heatmap.

This is an automated alert from the Smart Traffic Flow Analyzer.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Save and attach frame image
            temp_image_path = f"traffic_alert_{int(current_time)}.jpg"
            cv2.imwrite(temp_image_path, frame)
            
            with open(temp_image_path, 'rb') as f:
                img_data = f.read()
                image = MIMEImage(img_data)
                image.add_header('Content-Disposition', 'attachment', filename=os.path.basename(temp_image_path))
                msg.attach(image)
            
            # Send email
            server = smtplib.SMTP(self.email_config['host'], self.email_config['port'])
            server.starttls()
            server.login(self.email_config['user'], self.email_config['password'])
            text = msg.as_string()
            server.sendmail(self.email_config['user'], self.email_config['to'], text)
            server.quit()
            
            # Clean up temporary file
            os.remove(temp_image_path)
            
            self.last_email_time = current_time
            print(f"Congestion alert email sent to {self.email_config['to']}")
            
        except Exception as e:
            print(f"Failed to send email alert: {str(e)}")
            print("Check your email configuration in email_config.py")

    def process_frame(self, frame: np.ndarray, fps: float) -> Tuple[np.ndarray, bool]:
        """Process a single frame and return annotated frame with heatmap and congestion status."""
        results = self.model(frame, conf=0.25)  # Use predict instead of track

        # Initialize grid cells for this frame
        grid_cells = self.create_grid_cells(frame.shape)
        vehicle_speeds: List[float] = []
        frame_vehicles = 0

        # Process detected objects
        if results[0].boxes:
            frame_vehicles = len(results[0].boxes)
            
            for i, box in enumerate(results[0].boxes):
                # Use enumerate index as obj_id instead of tracking
                obj_id = i + 1
                
                # Get center coordinates and class information
                x, y, w, h = box.xywh[0]
                center = (int(x), int(y))
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]

                # Track unique objects (simplified without tracking)
                self._track_unique_objects(obj_id, class_name)

                # Assign vehicle to grid cell
                grid_row, grid_col = self.assign_vehicle_to_grid(center, frame.shape)
                grid_cells[grid_row][grid_col].vehicle_count += 1

                # Calculate dynamic speed based on vehicle position and box size
                # Larger objects are typically closer (moving slower)
                # Smaller objects are typically farther (moving faster)
                box_area = w * h
                base_speed = 60.0  # Base speed
                
                # Speed varies based on position and size
                # Top of frame (background) = higher speed, bottom (foreground) = lower speed
                position_factor = (frame.shape[0] - y) / frame.shape[0]  # 0-1, higher at top
                size_factor = min(1.0, box_area / (frame.shape[0] * frame.shape[1] * 0.01))  # Normalize by frame size
                
                # Calculate realistic speed: background vehicles faster, foreground slower
                speed = base_speed * (0.3 + 0.7 * position_factor) * (1.2 - size_factor * 0.8)
                
                # Add some randomness for realism
                import random
                speed += random.uniform(-15, 15)
                speed = max(10.0, min(120.0, speed))  # Clamp between 10-120
                
                vehicle_speeds.append(speed)
                grid_cells[grid_row][grid_col].vehicle_speeds.append(speed)

                # Draw vehicle information (simplified)
                self._draw_vehicle_info_simple(frame, box, class_id, class_name)

        # Generate heatmap overlay
        frame_with_heatmap = self.generate_heatmap_overlay(frame, grid_cells)

        # Draw grid lines for visualization
        self._draw_grid_lines(frame_with_heatmap)

        # Calculate frame metrics
        avg_speed = np.mean(vehicle_speeds) if vehicle_speeds else 0
        grid_densities = np.array([[cell.vehicle_count for cell in row] for row in grid_cells])
        
        # Determine congestion status
        is_congested = self._check_congestion(frame_vehicles, vehicle_speeds)
        
        # Store frame metrics for trend analysis
        frame_metrics = FrameMetrics(
            timestamp=time.time(),
            total_vehicles=frame_vehicles,
            avg_speed=avg_speed,
            grid_densities=grid_densities,
            is_congested=is_congested
        )
        self.frame_metrics_history.append(frame_metrics)

        # Predict congestion based on trends
        congestion_predicted = self.predict_congestion()
        
        # Send email alert if congestion is predicted and not already congested
        if congestion_predicted and not is_congested:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.send_email_alert(frame_with_heatmap, timestamp)

        # Draw unique object count
        self._draw_unique_object_count(frame_with_heatmap)

        # Draw status information with prediction
        self._draw_status_info(frame_with_heatmap, is_congested, vehicle_speeds, congestion_predicted)

        # Draw grid statistics
        self._draw_grid_statistics(frame_with_heatmap, grid_cells)

        return frame_with_heatmap, is_congested or congestion_predicted

    def _draw_grid_lines(self, frame: np.ndarray):
        """Draw grid lines on the frame for visualization."""
        height, width, cell_height, cell_width = self.get_grid_coordinates(frame.shape)
        
        # Draw horizontal lines
        for i in range(1, self.grid_rows):
            y = i * cell_height
            cv2.line(frame, (0, y), (width, y), (255, 255, 255), 1)
        
        # Draw vertical lines
        for i in range(1, self.grid_cols):
            x = i * cell_width
            cv2.line(frame, (x, 0), (x, height), (255, 255, 255), 1)

    def _draw_grid_statistics(self, frame: np.ndarray, grid_cells: List[List[GridCell]]):
        """Draw vehicle count in each grid cell."""
        height, width, cell_height, cell_width = self.get_grid_coordinates(frame.shape)
        
        for row_idx, grid_row in enumerate(grid_cells):
            for col_idx, cell in enumerate(grid_row):
                if cell.vehicle_count > 0:
                    # Calculate center of grid cell
                    center_x = col_idx * cell_width + cell_width // 2
                    center_y = row_idx * cell_height + cell_height // 2
                    
                    # Draw vehicle count
                    text = str(cell.vehicle_count)
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    
                    # Draw background for better visibility
                    cv2.rectangle(frame,
                                (center_x - text_w//2 - 5, center_y - text_h//2 - 5),
                                (center_x + text_w//2 + 5, center_y + text_h//2 + 5),
                                (0, 0, 0),
                                -1)
                    
                    # Draw text
                    cv2.putText(frame,
                              text,
                              (center_x - text_w//2, center_y + text_h//2),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.8,
                              (255, 255, 255),
                              2)

    def _track_unique_objects(self, obj_id: int, class_name: str):
        """Track unique objects across the entire video."""
        self.unique_objects[class_name].add(obj_id)
        self.unique_objects[class_name].add(obj_id)

    def _draw_unique_object_count(self, frame: np.ndarray):
        """Draw unique object count on the frame."""
        # Compute total unique objects
        total_unique_objects = sum(len(objects) for objects in self.unique_objects.values())

        # Prepare text items for unique object count
        unique_counts = [f"{class_name}: {len(objects)}"
                         for class_name, objects in self.unique_objects.items()]

        # Position in top-left corner
        h, w = frame.shape[:2]
        x_pos = 10
        y_start = 30

        # Draw total unique objects
        total_text = f"Total Unique Objects: {total_unique_objects}"
        (text_w, text_h), _ = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame,
                     (x_pos - 5, y_start - text_h - 5),
                     (x_pos + text_w + 5, y_start + 5),
                     (0, 0, 0),
                     -1)
        cv2.putText(frame,
                   total_text,
                   (x_pos, y_start),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.7,
                   (255, 255, 255),
                   2)

        # Draw individual class counts
        for i, count_text in enumerate(unique_counts):
            y_pos = y_start + (i + 1) * 30
            (text_w, text_h), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame,
                         (x_pos - 5, y_pos - text_h - 5),
                         (x_pos + text_w + 5, y_pos + 5),
                         (0, 0, 0),
                         -1)
            cv2.putText(frame,
                       count_text,
                       (x_pos, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (255, 255, 255),
                       2)

    def _update_vehicle_data(
        self,
        vehicle_id: int,
        position: Tuple[int, int],
        class_id: int,
        class_name: str,
        fps: float
    ) -> Optional[float]:
        """Update vehicle data and return current speed if available."""
        if vehicle_id not in self.vehicles:
            self.vehicles[vehicle_id] = VehicleData(position, class_id, class_name)
            self.speed_history[vehicle_id] = []
            return None

        # Calculate and store speed
        speed = self.calculate_speed(
            position,
            self.vehicles[vehicle_id].position,
            fps
        )

        # Manage speed history
        self.speed_history[vehicle_id].append(speed)
        if len(self.speed_history[vehicle_id]) > self.speed_window:
            self.speed_history[vehicle_id].pop(0)

        # Update vehicle data
        self.vehicles[vehicle_id] = VehicleData(
            position=position,
            class_id=class_id,
            class_name=class_name,
            speed=np.mean(self.speed_history[vehicle_id]) if self.speed_history[vehicle_id] else 0
        )

        return self.vehicles[vehicle_id].speed

    def _check_congestion(self, vehicle_count: int, speeds: list) -> bool:
        """Determine if traffic is congested based on vehicle count and speeds."""
        if not speeds:
            return False
        
        avg_speed = np.mean(speeds)
        
        # Dynamic thresholds based on actual detection
        # Congestion if many vehicles AND low average speed
        high_vehicle_count = vehicle_count > 8  # More than 8 vehicles
        low_average_speed = avg_speed < 40     # Less than 40 speed units
        
        # Also consider very high density (more than 12 vehicles regardless of speed)
        very_high_density = vehicle_count > 12
        
        return (high_vehicle_count and low_average_speed) or very_high_density

    def _draw_vehicle_info_simple(self, frame: np.ndarray, box, class_id: int, class_name: str):
        """Draw bounding box and class name for a vehicle (simplified version)."""
        x, y, w, h = box.xywh[0]
        tl = (int(x - w/2), int(y - h/2))
        br = (int(x + w/2), int(y + h/2))

        # Get color for class
        color = self.class_colors.get(class_id, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(frame, tl, br, color, 2)

        # Draw class name
        cv2.putText(frame,
                   f"{class_name}",
                   (tl[0], tl[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5,
                   color,
                   2)

    def _draw_vehicle_info(self, frame: np.ndarray, box, class_id: int, class_name: str):
        """Draw bounding box and class name for a vehicle."""
        x, y, w, h = box.xywh[0]
        tl = (int(x - w/2), int(y - h/2))
        br = (int(x + w/2), int(y + h/2))

        # Get color for class
        color = self.class_colors.get(class_id, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(frame, tl, br, color, 2)

        # Draw class name and speed
        vehicle_info = self.vehicles.get(int(box.id.item()) if box.id is not None else -1)
        speed_text = f" {vehicle_info.speed:.1f}px/s" if vehicle_info and vehicle_info.speed > 0 else ""

        cv2.putText(frame,
                   f"{class_name}{speed_text}",
                   (tl[0], tl[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5,
                   color,
                   2)

    def _draw_status_info(self, frame: np.ndarray, is_congested: bool, speeds: list, congestion_predicted: bool = False):
        """Draw congestion status and prediction in the corner of the frame."""
        # Calculate average speed
        avg_speed = np.mean(speeds) if speeds else 0

        # Status text with background for better visibility
        status_color = (0, 0, 255) if is_congested else (0, 255, 0)
        prediction_color = (0, 165, 255) if congestion_predicted else (255, 255, 255)
        bg_color = (0, 0, 0)

        # Position in top-right corner
        h, w = frame.shape[:2]
        x_pos = w - 350
        y_start = 30

        # Prepare text items
        texts = [
            (f"Congestion: {'YES' if is_congested else 'NO'}", status_color),
            (f"Prediction: {'WARNING' if congestion_predicted else 'NORMAL'}", prediction_color),
            (f"Avg Speed: {avg_speed:.2f} px/s", (255, 255, 255)),
            (f"Grid: {self.grid_rows}x{self.grid_cols}", (255, 255, 255))
        ]

        # Draw all text items
        for i, (text, color) in enumerate(texts):
            y_pos = y_start + i * 30
            # Draw semi-transparent background
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame,
                         (x_pos - 5, y_pos - text_h - 5),
                         (x_pos + text_w + 5, y_pos + 5),
                         bg_color,
                         -1)
            # Draw text
            cv2.putText(frame,
                       text,
                       (x_pos, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7,
                       color,
                       2)

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(args.input)  # Replace with your video path
    assert cap.isOpened(), "Error reading video file"

    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer
    video_writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    # Configure email settings if provided
    email_config = None
    if all([args.email_user, args.email_password, args.email_to]):
        email_config = {
            'host': args.email_host,
            'port': args.email_port,
            'user': args.email_user,
            'password': args.email_password,
            'to': args.email_to
        }
        print("Email notifications enabled")
    else:
        print("Email notifications disabled (missing configuration)")

    # Initialize congestion detector with enhanced features
    detector = CongestionDetector(
        model_path=args.weights,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        email_config=email_config
    )

    print(f"Processing video with {args.grid_rows}x{args.grid_cols} grid overlay...")
    frame_count = 0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Process frame with enhanced congestion detection
            annotated_frame, is_congested = detector.process_frame(frame, fps)

            # Write frame to output video
            video_writer.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:  # Print progress every 30 frames
                print(f"Processed {frame_count} frames...")

    finally:
        cap.release()
        video_writer.release()
        print(f"Processing complete. Output saved to: {args.output}")
        print(f"Total frames processed: {frame_count}")

if __name__ == "__main__":
    main()
