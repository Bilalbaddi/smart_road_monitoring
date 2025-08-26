# Quick Start Guide for Smart Traffic Flow Analyzer

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Python 3.8+
- Git installed
- Internet connection for model download

### Step 1: Clone and Setup
```bash
git clone https://github.com/Bilalbaddi/smart_road_monitoring.git
cd smart_road_monitoring
pip install -r requirements.txt
```

### Step 2: Download YOLO Model
The YOLO v11 model (~109MB) will be automatically downloaded on first run.
You can also manually download it:
```bash
# Create model directory
mkdir -p "Model yolo11x"
mkdir -p Congestion_detection

# Download model (choose one location)
# For Congestion_detection folder:
curl -L "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11x.pt" -o "Congestion_detection/yolo11x.pt"

# OR for Model yolo11x folder:
curl -L "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11x.pt" -o "Model yolo11x/yolo11x.pt"
```

### Step 3: Get Sample Video (Optional)
```bash
# Add your own traffic video as traffic_video.mp4
# Or download a sample (replace with actual sample video URL)
# curl -L "YOUR_SAMPLE_VIDEO_URL" -o "Congestion_detection/traffic_video.mp4"
```

### Step 4: Run Basic Analysis
```bash
cd Congestion_detection
python main.py
```

## 📧 Enable Email Alerts (Optional)

### Setup Email Configuration
```bash
# Copy template
cp email_config_template.py email_config.py

# Edit email_config.py with your settings
```

### Gmail Setup (Recommended)
1. Go to [Google Account Security](https://myaccount.google.com/security)
2. Enable 2-factor authentication
3. Go to [App Passwords](https://myaccount.google.com/apppasswords)
4. Generate app password for "Mail"
5. Edit `email_config.py`:
   ```python
   gmail_config = {
       'host': 'smtp.gmail.com',
       'port': 587,
       'user': 'your_email@gmail.com',
       'password': 'your_app_password',  # Use the generated app password
       'to': 'recipient@gmail.com'
   }
   return gmail_config  # Change this line
   ```

## ⚡ Advanced Usage

### Custom Grid Size
```python
# Edit main.py to change grid configuration
detector = CongestionDetector(
    grid_size=(8, 8),  # Change to (4,4) or (6,6) etc.
    congestion_threshold=15
)
```

### Different Video Input
```bash
python main.py --input your_video.mp4 --output analysis_output.mp4
```

## 🔧 Troubleshooting

### Issue: ModuleNotFoundError
```bash
# Solution: Install missing dependencies
pip install ultralytics opencv-python numpy
```

### Issue: CUDA/GPU not working
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA version of PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Model download fails
```bash
# Manual download with wget (Linux/Mac) or curl
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11x.pt
# or
curl -L "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11x.pt" -o "yolo11x.pt"
```

### Issue: Email not sending
1. Check your email credentials
2. Verify app password (for Gmail/Yahoo)
3. Test configuration:
   ```bash
   python email_config_template.py
   ```

## 📊 Expected Output

After running the analysis, you'll get:
- **Processed Video**: With vehicle detection, speed estimates, and grid overlay
- **Console Output**: Real-time statistics and alerts
- **Email Notifications**: If configured and congestion detected

### Sample Console Output:
```
Processing frame 100/500...
Detected vehicles: 12
Average speed: 45.2
Grid congestion: [2, 5, 3, 1, 4, 7, 2, 1, 0, 3, 2, 1, 1, 0, 2, 1]
Congestion detected in zones: [1, 5]
Email alert sent successfully!
```

## 🎯 Next Steps

1. **Experiment with Settings**: Try different grid sizes and thresholds
2. **Analyze Your Videos**: Use your own traffic footage
3. **Set Up Monitoring**: Configure email alerts for automated monitoring
4. **Contribute**: Check the main README for contribution guidelines

## 📞 Support

- **Issues**: Open an issue on GitHub
- **Email**: bilalbaddi074@gmail.com
- **Documentation**: See main README.md

---
Happy analyzing! 🚗📊