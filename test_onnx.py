import onnxruntime as ort
import cv2
import numpy as np
import os

session_model_1 = ort.InferenceSession(r"C:\Users\vladi\Downloads\best(1).onnx")
session_model_2 = ort.InferenceSession(r"C:\Users\vladi\Downloads\people320_25.onnx")

input_name_model_1 = session_model_1.get_inputs()[0].name
output_names_model_1 = [output.name for output in session_model_1.get_outputs()]

input_name_model_2 = session_model_2.get_inputs()[0].name
output_names_model_2 = [output.name for output in session_model_2.get_outputs()]

image_folder = r"C:\Users\vladi\Downloads\TestOnnxImages"
images = [img for img in os.listdir(image_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

for image_name in images:
    image_path = os.path.join(image_folder, image_name)
    image_np = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (224, 224))
    input_tensor = np.expand_dims(image_resized, axis=0).astype(np.float32)

    outputs = session_model_1.run(output_names_model_1, {input_name_model_1: input_tensor})
    print(len(outputs[0][0][12]))
    detection_boxes_model_1, detection_classes_model_1, detection_scores_model_1, num_detections_model_1 = (
        outputs_model_1[1][0], outputs_model_1[2][0], outputs_model_1[4][0], int(outputs_model_1[5][0])
    )
    detection_boxes_with_classes = np.column_stack((detection_boxes_model_1, detection_classes_model_1))
    print(detection_boxes_with_classes)

    h, w = image_np.shape[:2]
    detections_model_1 = [
        {
            "box": [int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)],
            "score": detection_scores_model_1[i],
            "class": detection_classes_model_1[i]
        }
        for i in range(num_detections_model_1)
        for ymin, xmin, ymax, xmax in [detection_boxes_model_1[i]]
        if detection_scores_model_1[i] >= 0.25
    ]

    outputs_model_2 = session_model_2.run(output_names_model_2, {input_name_model_2: input_tensor})
    detection_boxes_model_2, detection_classes_model_2, detection_scores_model_2, num_detections_model_2 = (
        outputs_model_2[1][0], outputs_model_2[2][0], outputs_model_2[4][0], int(outputs_model_2[5][0])
    )

    detections_model_2 = [
        {
            "box": [int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)],
            "score": detection_scores_model_2[i],
            "class": detection_classes_model_2[i]
        }
        for i in range(num_detections_model_2)
        for ymin, xmin, ymax, xmax in [detection_boxes_model_2[i]]
        if detection_scores_model_2[i] >= 0.25
    ]

    for det in detections_model_1:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (20, 255, 0), 2)  # Зеленый цвет для первой модели
        cv2.putText(image_np, f"{det['score']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

    for det in detections_model_2:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 200), 2)  # Синий цвет для второй модели
        cv2.putText(image_np, f"{det['score']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 0), 2)

    cv2.imshow(f"Detection - {image_name}", image_np)
    cv2.waitKey(0)

cv2.destroyAllWindows()
