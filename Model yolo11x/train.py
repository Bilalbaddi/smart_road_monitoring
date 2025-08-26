import os
HOME = os.getcwd()
print(HOME)

!pip install -q roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="Your API Key")
project = rf.workspace("dreamfalls").project("street-view-gdogo-bvynj")
version = project.version(1)
dataset = version.download("yolov11")

dataset_path = dataset.location

dataset_path

# Commented out IPython magic to ensure Python compatibility.
# %pip install ultralytics  # install
from ultralytics import YOLO, checks, hub
checks()  # checks

!pip install -U ipywidgets

!pip install google-cloud-bigquery[bqstorage,pandas]==3.10.0
!pip install google-cloud-storage==2.0.0
!pip install pandas==1.5.0  # Ensure to select a version below 2.1.4
!pip install pydantic==1.10.0

!pip install --upgrade jax jaxlib wandb

import wandb

# Log in with your API key
wandb.login(key='Your API Key')

# Step 3: Initialize YOLOv8-OBB model
model = YOLO('yolo11x.pt')  # Change to yolov8n.yaml, yolov8s.yaml, etc. for different model sizes

# Step 4: Train the model
model.train(data="/kaggle/input/data-yaml/data.yaml", imgsz=640, epochs=25, batch=16, device=[0, 1])# Adjust epochs, image size, batch size as needed

# Step 6: Save the trained model
model.save('yolo_best.pt')

model = YOLO('/kaggle/working/yolo_best.pt')
