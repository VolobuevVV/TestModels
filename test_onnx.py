import onnxruntime as ort
import cv2
import numpy as np

session = ort.InferenceSession("model512.onnx")

input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

image_path = r"C:\Users\vladi\Downloads\photo.jpg"
image_np = cv2.imread(image_path)
image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
image_np_resized = cv2.resize(image_np, (512, 512))
input_tensor = np.expand_dims(image_np_resized, axis=0).astype(np.uint8)

outputs = session.run(output_names, {input_name: input_tensor})
print(outputs)
detection_boxes = outputs[1][0]
detection_classes = outputs[2][0]
detection_scores = outputs[4][0]
num_detections = int(outputs[5][0])

class_labels = ["Class 0", "Class 1", "Class 2"]

for i in range(num_detections):
    score = detection_scores[i]
    if score > 0.5:
        box = detection_boxes[i]
        ymin, xmin, ymax, xmax = box
        xmin = int(xmin * image_np.shape[1])
        xmax = int(xmax * image_np.shape[1])
        ymin = int(ymin * image_np.shape[0])
        ymax = int(ymax * image_np.shape[0])

        cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

        class_id = int(detection_classes[i])
        label = f"{class_labels[class_id]}: {score:.2f}"
        cv2.putText(image_np, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

cv2.imshow('Detection Result', image_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
