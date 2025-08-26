Original Dataset:

- Version 1:
  - Kaggle: https://www.kaggle.com/datasets/yusufberksardoan/traffic-detection-project
  - Roboflow: https://universe.roboflow.com/fsmvu/street-view-gdogo/dataset/1

  Total Images: 6,633

  Dataset Split:
  - Train Set: 88% (5,805 Images)
  - Validation Set: 8% (549 Images)
  - Test Set: 4% (279 Images)

  Preprocessing:
  - Auto-Orient: Applied
  - Resize: Stretch to 640x640

  Augmentations (Outputs per training example: 3):
  - Flip: Horizontal
  - Saturation: Between -61% and +61%
  - Brightness: Between -25% and +25%

- Version 2:
  - Roboflow: https://universe.roboflow.com/dreamfalls/street-view-gdogo-bvynj/dataset/3

  Total Images: 8,693

  Dataset Split:
  - Train Set: 87% (7,566 Images)
  - Validation Set: 9% (805 Images)
  - Test Set: 4% (322 Images)

  Preprocessing:
  - Auto-Orient: Applied
  - Resize: Stretch to 640x640

  Augmentations (Outputs per training example: 3):
  - Flip: Horizontal
  - Saturation: Between -61% and +61%
  - Brightness: Between -25% and +25%
  - Noise: Up to 2% of pixels

Modified Dataset:

- Roboflow: https://universe.roboflow.com/dreamfalls/street-view-gdogo-bvynj/dataset/1
- Kaggle: https://www.kaggle.com/datasets/tsp1718/smart-traffic-flow-analyzer-dataset

  Total Images: 8,693

  Dataset Split:
  - Train Set: 87% (7,566 Images)
  - Validation Set: 9% (805 Images)
  - Test Set: 4% (322 Images)

  Preprocessing:
  - Auto-Orient: Applied
  - Resize: Stretch to 640x640

  Augmentations (Outputs per training example: 3):
  - Flip: Horizontal
  - Saturation: Between -61% and +61%
  - Brightness: Between -25% and +25%
  - Blur: Up to 1.2px
  - Noise: Up to 2% of pixels
  - Bounding Box Brightness: Between -25% and +25%
  - Bounding Box Blur: Up to 1.2px
