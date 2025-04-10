import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

model1_path = r"C:\Users\vladi\Downloads\g_11n_224_100ep.pt"
model2_path = r"C:\Users\vladi\Downloads\g_11n_224_150ep.pt"

model1 = YOLO(model1_path)
model2 = YOLO(model2_path)

image_folder = r"C:\Users\vladi\Downloads\test_gtrucks"

image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
               if os.path.splitext(f)[-1].lower() in image_extensions]

for image_path in image_paths:
    results1 = model1(image_path)
    results2 = model2(image_path)

    annotated_img1 = results1[0].plot()
    annotated_img2 = results2[0].plot()

    annotated_img1 = cv2.cvtColor(annotated_img1, cv2.COLOR_BGR2RGB)
    annotated_img2 = cv2.cvtColor(annotated_img2, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    axes[0].imshow(annotated_img1)
    axes[0].set_title(f'Model 1: {os.path.basename(model1_path)}')
    axes[0].axis('off')

    axes[1].imshow(annotated_img2)
    axes[1].set_title(f'Model 2: {os.path.basename(model2_path)}')
    axes[1].axis('off')

    plt.suptitle(os.path.basename(image_path))
    plt.show()
