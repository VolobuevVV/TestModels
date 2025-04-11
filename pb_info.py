import tensorflow as tf
import cv2

model_path = r"C:\Users\vladi\Downloads\custom_model_lite (1)\custom_model_lite\saved_model"
loaded_model = tf.saved_model.load(model_path)

image_path = r"C:\Users\vladi\Downloads\photo.jpg"
image_np = cv2.imread(image_path)
image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
image_np_resized = cv2.resize(image_np, (320, 320))

input_tensor = tf.convert_to_tensor(image_np_resized, dtype=tf.uint8)
input_tensor = input_tensor[tf.newaxis, ...]

infer = loaded_model.signatures['serving_default']
output_dict = infer(input_tensor)

detection_boxes = output_dict['detection_boxes'].numpy()
detection_classes = output_dict['detection_classes'].numpy()
detection_scores = output_dict['detection_scores'].numpy()

print("Detection boxes:", detection_boxes)
print("Detection classes:", detection_classes)
print("Detection scores:", detection_scores)

for i in range(len(detection_scores[0])):
    score = detection_scores[0][i]
    if score > 0.5:
        box = detection_boxes[0][i]
        (ymin, xmin, ymax, xmax) = box
        xmin = int(xmin * image_np.shape[1])
        xmax = int(xmax * image_np.shape[1])
        ymin = int(ymin * image_np.shape[0])
        ymax = int(ymax * image_np.shape[0])
        cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        class_id = int(detection_classes[0][i])
        score = detection_scores[0][i]
        label = f"Class {class_id}, Score: {score:.2f}"
        cv2.putText(image_np, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
cv2.imshow('Detection Result', image_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
