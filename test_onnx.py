import onnxruntime as ort
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

session = ort.InferenceSession("model.onnx")

input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

image_folder = r"TestImagesLarge"
images = os.listdir(image_folder)

fig, axes = plt.subplots(len(images), 1, figsize=(10, len(images)*5))

if len(images) == 1:
    axes = [axes]

for i, image_name in enumerate(images):
    image_path = os.path.join(image_folder, image_name)

    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        continue

    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_np, (320, 320))
    input_tensor = np.expand_dims(image_np, axis=0).astype(np.uint8)

    outputs = session.run(output_names, {input_name: input_tensor})

    detection_boxes = outputs[1][0]
    detection_classes = outputs[2][0]
    detection_scores = outputs[4][0]
    num_detections = int(outputs[5][0])

    for j in range(num_detections):
        score = detection_scores[j]
        if score > 0.5:
            box = detection_boxes[j]
            ymin, xmin, ymax, xmax = box
            xmin = int(xmin * image_np.shape[1])
            xmax = int(xmax * image_np.shape[1])
            ymin = int(ymin * image_np.shape[0])
            ymax = int(ymax * image_np.shape[0])

            cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    axes[i].imshow(image_np)
    axes[i].axis('off')
    axes[i].set_title(f"Detection Result {image_name}")

plt.tight_layout()
plt.show()
