import ultralytics

ultralytics.checks()

import cv2
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import argparse

# Define the argument parser for input and output paths
parser = argparse.ArgumentParser(description="Run congestion detection on a video.")
parser.add_argument("--weights", type=str, required=True, help="Path to the YOLO model weights")
parser.add_argument("--input", type=str, required=True, help="Path to the input video")
parser.add_argument("--output", type=str, default="output_video.mp4", help="Path to save the output video")

args = parser.parse_args()

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
        vehicle_threshold: int = 6,
        speed_threshold: float = 75,
        speed_window: int = 30
    ):
        self.model = YOLO(model_path)
        self.vehicle_threshold = vehicle_threshold
        self.speed_threshold = speed_threshold
        self.speed_window = speed_window

        # State tracking
        self.vehicles: Dict[int, VehicleData] = {}
        self.speed_history: Dict[int, list] = {}

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

    def process_frame(self, frame: np.ndarray, fps: float) -> Tuple[np.ndarray, bool]:
        """Process a single frame and return annotated frame and congestion status."""
        results = self.model.track(frame, persist=True, conf=0.25)

        vehicle_speeds: List[float] = []

        # Process detected objects
        if results[0].boxes:
            for box in results[0].boxes:
                obj_id = int(box.id.item()) if box.id is not None else None
                if obj_id is None:
                    continue

                # Get center coordinates and class information
                x, y, w, h = box.xywh[0]
                center = (int(x), int(y))
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]

                # Track unique objects
                self._track_unique_objects(obj_id, class_name)

                # Update or add vehicle data
                speed = self._update_vehicle_data(
                    obj_id,
                    center,
                    class_id,
                    class_name.lower(),
                    fps
                )

                if speed is not None:
                    vehicle_speeds.append(speed)

                # Draw vehicle information
                self._draw_vehicle_info(frame, box, class_id, class_name)

        # Draw unique object count
        self._draw_unique_object_count(frame)

        # Determine congestion status
        is_congested = self._check_congestion(len(results[0].boxes), vehicle_speeds)

        # Draw status information
        self._draw_status_info(frame, is_congested, vehicle_speeds)

        return frame, is_congested

    def _track_unique_objects(self, obj_id: int, class_name: str):
        """Track unique objects across the entire video."""
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
        return vehicle_count > self.vehicle_threshold and avg_speed < self.speed_threshold

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

    def _draw_status_info(self, frame: np.ndarray, is_congested: bool, speeds: list):
        """Draw congestion status in the corner of the frame."""
        # Calculate average speed
        avg_speed = np.mean(speeds) if speeds else 0

        # Status text with background for better visibility
        status_color = (0, 0, 255) if is_congested else (0, 255, 0)
        bg_color = (0, 0, 0)

        # Position in top-right corner
        h, w = frame.shape[:2]
        x_pos = w - 250
        y_start = 30

        # Prepare text items
        texts = [
            (f"Congestion: {'YES' if is_congested else 'NO'}", status_color),
            (f"Avg Speed: {avg_speed:.2f} px/s", (255, 255, 255))
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

    # Initialize congestion detector
    detector = CongestionDetector(args.weights)  # Replace with your model path

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Process frame
            annotated_frame, is_congested = detector.process_frame(frame, fps)

            # Write frame to output video
            video_writer.write(annotated_frame)


    finally:
        cap.release()
        video_writer.release()

if __name__ == "__main__":
    main()
