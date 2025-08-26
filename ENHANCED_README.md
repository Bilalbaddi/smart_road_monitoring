# Enhanced Smart Traffic Flow Analyzer

An advanced computer vision system for real-time traffic analysis with congestion detection, predictive analytics, and automated alerting capabilities.

## 🚀 New Features

### 1. Grid-Based Analysis
- **Configurable Grid Division**: Divide video frames into customizable grid cells (e.g., 4x4, 8x8)
- **Per-Cell Vehicle Counting**: Track vehicle density in each grid cell
- **Spatial Congestion Mapping**: Identify congestion hotspots within the monitoring area

### 2. Dynamic Heatmap Visualization
- **Real-Time Heatmap Overlay**: Transparent color-coded overlay showing vehicle density
- **Color-Coded Density Levels**:
  - 🟢 **Green**: Low density (< 30% of maximum)
  - 🟡 **Yellow/Orange**: Medium density (30-70% of maximum)
  - 🔴 **Red**: High density (> 70% of maximum)
- **Transparent Overlay**: Maintains video visibility while showing congestion patterns

### 3. Predictive Congestion Analytics
- **Trend Analysis**: Tracks vehicle speed and density over the last 5-10 frames
- **Early Warning System**: Predicts congestion before it becomes critical
- **Smart Algorithms**: Detects when average speed decreases while density increases

### 4. Automated Email Notifications
- **SMTP Integration**: Sends email alerts when congestion is predicted
- **Rich Notifications**: Includes timestamp and heatmap screenshot
- **Configurable Cooldown**: Prevents spam with adjustable alert intervals
- **Multiple Provider Support**: Works with Gmail, Outlook, and other SMTP providers

### 5. Enhanced Visualization
- **Grid Overlay**: Visual grid lines showing analysis zones
- **Real-Time Statistics**: Live vehicle counts per grid cell
- **Status Dashboard**: Shows current congestion status and predictions
- **Progress Tracking**: Frame processing progress indicators

## 📋 Requirements

```bash
pip install -r requirements.txt
```

**Dependencies:**
- ultralytics (YOLO v11)
- opencv-python
- numpy

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Smart-Traffic-Flow-Analyzer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download YOLO model weights:**
Place your YOLO model weights in the `Model yolo11x/` directory.

## 🚦 Usage

### Basic Usage

```bash
python main.py --weights path/to/model.pt --input traffic_video.mp4 --output output.mp4
```

### Advanced Usage with Grid Configuration

```bash
python main.py \
    --weights Model\ yolo11x/yolo11x.pt \
    --input Congestion_detection/traffic_video.mp4 \
    --output enhanced_output.mp4 \
    --grid-rows 8 \
    --grid-cols 8
```

### With Email Notifications

```bash
python main.py \
    --weights Model\ yolo11x/yolo11x.pt \
    --input Congestion_detection/traffic_video.mp4 \
    --output output_with_alerts.mp4 \
    --grid-rows 6 \
    --grid-cols 6 \
    --email-user your-email@gmail.com \
    --email-password your-app-password \
    --email-to recipient@example.com \
    --email-host smtp.gmail.com \
    --email-port 587
```

## ⚙️ Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--weights` | Path to YOLO model weights | Required |
| `--input` | Input video file path | Required |
| `--output` | Output video file path | `output_video.mp4` |
| `--grid-rows` | Number of horizontal grid divisions | `4` |
| `--grid-cols` | Number of vertical grid divisions | `4` |
| `--email-host` | SMTP server hostname | `smtp.gmail.com` |
| `--email-port` | SMTP server port | `587` |
| `--email-user` | Sender email address | None |
| `--email-password` | Sender email password | None |
| `--email-to` | Recipient email address | None |

## 📧 Email Setup

### Gmail Configuration
1. Enable 2-factor authentication
2. Generate an app-specific password
3. Use the app password instead of your regular password

### Other Providers
- **Outlook/Hotmail**: `smtp.outlook.com:587`
- **Yahoo**: `smtp.mail.yahoo.com:587`
- **Custom SMTP**: Specify your provider's settings

## 🏗️ Architecture

### Core Components

#### `CongestionDetector` Class
- **Grid Management**: Creates and manages grid cells
- **Vehicle Tracking**: Enhanced YOLO-based detection with grid assignment
- **Heatmap Generation**: Real-time density visualization
- **Prediction Engine**: Trend analysis for early congestion detection
- **Alert System**: Email notification management

#### Key Methods

- `generate_heatmap_overlay()`: Creates transparent density heatmap
- `predict_congestion()`: Analyzes trends for early warning
- `send_email_alert()`: Handles automated notifications
- `assign_vehicle_to_grid()`: Maps vehicles to grid cells

### Data Structures

```python
@dataclass
class GridCell:
    row: int
    col: int
    vehicle_count: int = 0
    vehicle_speeds: List[float] = None

@dataclass
class FrameMetrics:
    timestamp: float
    total_vehicles: int
    avg_speed: float
    grid_densities: np.ndarray
    is_congested: bool
```

## 🎯 Use Cases

### Traffic Management
- **Real-time monitoring** of traffic flow
- **Hotspot identification** for traffic optimization
- **Incident detection** and response

### Urban Planning
- **Data collection** for infrastructure planning
- **Pattern analysis** for traffic light optimization
- **Capacity planning** for road improvements

### Smart City Integration
- **Automated reporting** to traffic control centers
- **API integration** with city management systems
- **Historical data** for trend analysis

## 📊 Output Features

### Video Output
- **Enhanced visualization** with heatmap overlay
- **Grid annotations** showing analysis zones
- **Real-time statistics** overlay
- **Status indicators** for congestion and predictions

### Email Alerts
- **Timestamp information**
- **Congestion severity details**
- **Visual confirmation** with heatmap screenshot
- **Actionable insights** for traffic management

## 🔧 Customization

### Adjust Detection Sensitivity
Modify thresholds in the `CongestionDetector` constructor:
```python
detector = CongestionDetector(
    model_path="model.pt",
    vehicle_threshold=6,      # Minimum vehicles for congestion
    speed_threshold=75,       # Maximum speed for congestion (px/s)
    frame_history_size=10     # Frames for trend analysis
)
```

### Customize Colors
Edit the heatmap colors in the `generate_heatmap_overlay()` method:
```python
# Low density: Green
# Medium density: Yellow/Orange  
# High density: Red
```

### Email Cooldown
Adjust notification frequency:
```python
self.email_cooldown = 300  # 5 minutes between emails
```

## 🚨 Troubleshooting

### Common Issues

1. **Email Authentication Failures**
   - Ensure 2FA is enabled
   - Use app-specific passwords
   - Check SMTP settings

2. **Model Loading Errors**
   - Verify model path is correct
   - Ensure YOLO weights are compatible
   - Check file permissions

3. **Video Processing Issues**
   - Verify video codec compatibility
   - Check input file format
   - Ensure sufficient disk space

4. **Performance Optimization**
   - Reduce grid resolution for faster processing
   - Lower video resolution if needed
   - Adjust detection confidence threshold

## 📈 Performance Metrics

- **Real-time processing** capability (depending on hardware)
- **Scalable grid resolution** (4x4 to 16x16 tested)
- **Memory efficient** tracking algorithms
- **Configurable processing** parameters

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement enhancements
4. Add comprehensive tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Projects

- [YOLO v11](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [Traffic Analysis Research](https://scholar.google.com/scholar?q=traffic+congestion+detection)

## ✨ Acknowledgments

- YOLO team for object detection framework
- OpenCV community for computer vision tools
- Contributors to traffic analysis research

---

**Built with ❤️ for smarter traffic management**