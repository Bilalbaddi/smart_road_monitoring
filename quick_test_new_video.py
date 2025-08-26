#!/usr/bin/env python3
"""
Quick test script for the new traffic video
Processes only the first 30 frames to demonstrate functionality
"""

import cv2
import numpy as np
from main import CongestionDetector
import os
import sys

def quick_test_new_video():
    """Test the enhanced system on the first 30 frames of new video."""
    print("🚗 Quick Test: Enhanced Traffic Analysis on New Video")
    print("=" * 55)
    
    # Configuration
    model_path = "yolo11x.pt"
    input_video = "new_traffic_video.mp4"
    output_video = "new_video_quick_test.mp4"
    max_frames = 30  # Process only first 30 frames (1 second at 30 FPS)
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        return
    
    if not os.path.exists(input_video):
        print(f"❌ Error: Video file '{input_video}' not found!")
        return
    
    print(f"📹 Input Video: {input_video}")
    print(f"🤖 Model: {model_path}")
    print(f"📊 Grid Configuration: 4x4")
    print(f"⏱️ Processing: First {max_frames} frames only")
    print(f"💾 Output: {output_video}")
    print()
    
    try:
        # Initialize detector
        detector = CongestionDetector(
            model_path=model_path,
            grid_rows=4,
            grid_cols=4,
            vehicle_threshold=8,  # Lower threshold for demo
            speed_threshold=40,
            frame_history_size=10
        )
        
        # Open video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print("❌ Error: Could not open video file!")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📋 Video Info:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total Frames: {total_frames}")
        print(f"   Duration: {total_frames/fps:.1f} seconds")
        print()
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        # Initialize grid
        grid_cells = detector.create_grid_cells((height, width, 3))
        frame_count = 0
        vehicle_detections = []
        
        print("🔄 Processing frames...")
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run YOLO detection
            results = detector.model(frame, verbose=False)
            
            # Process detections
            vehicles_in_frame = 0
            speeds_in_frame = []
            
            # Reset grid cells
            for row in grid_cells:
                for cell in row:
                    cell.vehicle_count = 0
                    cell.vehicle_speeds = []
            
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                
                for box in boxes:
                    # Get vehicle info
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Filter for vehicles with high confidence
                    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
                    if class_id in vehicle_classes and confidence > 0.5:
                        # Get bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        # Calculate dynamic speed (enhanced algorithm)
                        box_area = (x2 - x1) * (y2 - y1)
                        base_speed = 60.0
                        position_factor = (height - center_y) / height
                        size_factor = min(1.0, box_area / (height * width * 0.01))
                        speed = base_speed * (0.3 + 0.7 * position_factor) * (1.2 - size_factor * 0.8)
                        speed = max(10.0, min(120.0, speed))  # Clamp 10-120
                        
                        # Assign to grid
                        grid_row, grid_col = detector.assign_vehicle_to_grid((center_x, center_y), (height, width, 3))
                        grid_cells[grid_row][grid_col].vehicle_count += 1
                        grid_cells[grid_row][grid_col].vehicle_speeds.append(speed)
                        
                        vehicles_in_frame += 1
                        speeds_in_frame.append(speed)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"{speed:.0f}", (int(x1), int(y1-10)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Generate heatmap overlay
            frame_with_heatmap = detector.generate_heatmap_overlay(frame, grid_cells)
            
            # Add frame statistics
            avg_speed = np.mean(speeds_in_frame) if speeds_in_frame else 0
            cv2.putText(frame_with_heatmap, f"Frame: {frame_count}/{max_frames}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame_with_heatmap, f"Vehicles: {vehicles_in_frame}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame_with_heatmap, f"Avg Speed: {avg_speed:.1f}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Check for congestion
            congested = vehicles_in_frame > 8 and avg_speed < 40
            if congested:
                cv2.putText(frame_with_heatmap, "CONGESTION DETECTED!", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print(f"   🚨 Frame {frame_count}: Congestion detected! {vehicles_in_frame} vehicles, {avg_speed:.1f} avg speed")
            
            # Save detection info
            vehicle_detections.append({
                'frame': frame_count,
                'vehicles': vehicles_in_frame,
                'avg_speed': avg_speed,
                'congested': congested
            })
            
            # Write frame
            out.write(frame_with_heatmap)
            
            # Progress indicator
            if frame_count % 10 == 0:
                print(f"   Processed {frame_count}/{max_frames} frames...")
        
        # Cleanup
        cap.release()
        out.release()
        
        print("\n✅ Quick test completed!")
        print(f"📊 Analysis Summary:")
        
        total_vehicles = sum(d['vehicles'] for d in vehicle_detections)
        avg_vehicles_per_frame = total_vehicles / len(vehicle_detections)
        congested_frames = sum(1 for d in vehicle_detections if d['congested'])
        avg_speed_overall = np.mean([d['avg_speed'] for d in vehicle_detections if d['avg_speed'] > 0])
        
        print(f"   Total vehicles detected: {total_vehicles}")
        print(f"   Average vehicles per frame: {avg_vehicles_per_frame:.1f}")
        print(f"   Congested frames: {congested_frames}/{frame_count}")
        print(f"   Overall average speed: {avg_speed_overall:.1f}")
        print(f"   Output video: {output_video}")
        
        # Show most interesting frames
        max_vehicles_frame = max(vehicle_detections, key=lambda x: x['vehicles'])
        print(f"\n🎯 Most traffic: Frame {max_vehicles_frame['frame']} with {max_vehicles_frame['vehicles']} vehicles")
        
        if congested_frames > 0:
            first_congestion = next(d for d in vehicle_detections if d['congested'])
            print(f"🚨 First congestion: Frame {first_congestion['frame']}")
        
        print(f"\n🎬 To view the result, open: {output_video}")
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test_new_video()