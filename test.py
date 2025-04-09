import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

class TFLiteModel:
    def __init__(self, modelpath):
        self.interpreter = Interpreter(model_path=modelpath)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.float_input = (self.input_details[0]['dtype'] == np.float32)
        self.input_mean = 127.5
        self.input_std = 127.5

    def detect(self, image, min_conf=0.2):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        image_resized = cv2.resize(image_rgb, (self.width, self.height))
        input_data = np.expand_dims(image_resized, axis=0)

        if self.float_input:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[3]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        detections = []

        for i in range(len(scores)):
            if (scores[i] > min_conf) and (scores[i] <= 1.0):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                object_name = str(classes[i])
                label = '%s: %.2f%%' % (object_name, scores[i] * 100)

                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10),
                              (xmin + labelSize[0], label_ymin + baseLine - 10),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

        return image

def handle_image(update: Update, context: CallbackContext):
    photo_file = update.message.photo[-1].get_file()
    photo_path = "received_image.jpg"
    photo_file.download(photo_path)

    detected_image = model.detect(cv2.imread(photo_path), min_conf=0.5)

    output_path = "output_image.jpg"
    cv2.imwrite(output_path, detected_image)

    with open(output_path, 'rb') as photo:
        update.message.reply_photo(photo)

def handle_non_photo(update: Update, context: CallbackContext):
    update.message.reply_text("Пожалуйста, отправьте изображение")

def main():
    global model
    model = TFLiteModel("detect2.tflite")

    TOKEN = 'YOUR_BOT_TOKEN'

    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(MessageHandler(Filters.photo, handle_image))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_non_photo))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
