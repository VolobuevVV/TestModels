import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

model1_path = r"C:\Users\vladi\Downloads\best(1).onnx"

model1 = YOLO(model1_path)

image_folder = r"C:\Users\vladi\Downloads\TestPeopleCounting"

image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
               if os.path.splitext(f)[-1].lower() in image_extensions]

for image_path in image_paths:
    image_np = cv2.imread(image_path)
    image_resized = cv2.resize(image_np, (224, 224))
    results1 = model1(image_np, conf=0.2)
    print(results1)

    annotated_img1 = results1[0].plot()

    annotated_img1 = cv2.cvtColor(annotated_img1, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    axes[0].imshow(annotated_img1)
    axes[0].set_title(f'Model 1: {os.path.basename(model1_path)}')
    axes[0].axis('off')

    plt.suptitle(os.path.basename(image_path))
    plt.show()
