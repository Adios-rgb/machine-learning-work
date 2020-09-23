import cv2
import time
import numpy as np
import imutils
import keras
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet201


protoPath = "path to deploy.prototxt.txt"
modelPath = "path to res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch('path to openface.nn4.small2.v1.t7')

# This method is used to capture images from your webcam
def capture_data_from_videocapture():
	cap = cv2.VideoCapture(0)
	count = 0
	pics_count = count + 30
	time.sleep(2)
	while True:
		count += 1
		_, frame = cap.read()
		frame = imutils.resize(frame, width=400)
		h, w = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
		detector.setInput(blob)
		detections = detector.forward()

		if len(detections) > 0:
			for i in range(0, detections.shape[2]):
				try:
					confidence = detections[0, 0, i, 2]
					if confidence > 0.6:
						box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
						(startX, startY, endX, endY) = box.astype("int")
						cv2.putText(frame, 'Face Detected', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
						frame_copy = frame.copy()
						cv2.rectangle(frame_copy, (startX, startY), (endX, endY), (0, 0, 255), 2)
						face = frame[startY - 5: endY + 5, startX - 5: endX + 5]
						face_expression = cv2.resize(face, (96, 96))
				except Exception as e:
					print(str(e))
					continue
			cv2.imwrite(f'path to output images image_name_{count}.png', face_expression)
			time.sleep(2)
		if pics_count - count == 0:
			break
		cv2.imshow('frame', frame_copy)
		cv2.waitKey(1)
	cap.release()
	cv2.destroyAllWindows()


pretrained_model = keras.applications.DenseNet201(weights='imagenet', include_top=False ,input_shape=(96, 96, 3))
pretrained_model.trainable = False # False = transfer learning, True = fine-tuning
    
x = pretrained_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(2, activation='softmax')(x)

densenet_model = Model(pretrained_model.input, predictions)
# Do not forget to compile it
densenet_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

image_gen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1/255.0,
    shear_range=0.1,
    zoom_range=0.2,
    fill_mode='nearest'
)

test_data = ImageDataGenerator(
    rescale=1/255.0,
)

# Notice the tiny target size, just 48x48!
train_set = image_gen.flow_from_directory(
    r'path to images for training',
    target_size=(96, 96),
    batch_size=64
)
test_set = test_data.flow_from_directory(
    r'path to images for testing',
    target_size=(96, 96),
    batch_size=64
)

checkpoint = ModelCheckpoint('path to save best model model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
hist = densenet_model.fit_generator(train_set, steps_per_epoch=35, epochs=50, validation_data=test_set, validation_steps=20, callbacks=[
        checkpoint,
        EarlyStopping(patience=4, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(patience=3)
    ])


# Training the model again with very small learning rate and this time making pretrained maodel as trainable
pretrained_model.trainable = True
densenet_model = keras.models.load_model('path to saved model model.h5')
densenet_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=keras.optimizers.Adam(0.00002))
checkpoint = ModelCheckpoint('path to save the new best model model2.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
hist = densenet_model.fit_generator(train_set, steps_per_epoch=35, epochs=50, validation_data=test_set, validation_steps=20, callbacks=[
        checkpoint,
        EarlyStopping(patience=3, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(patience=2)
    ])
	

model = keras.models.load_model('path to trained model model.h5')
def detect_expression(model):
    cap = cv2.VideoCapture(0)
    time.sleep(2)
    t1 = time.time()
    while True:
        t2 = time.time()
        _, frame = cap.read()
        frame = imutils.resize(frame, width=400)
        frame_copy = frame.copy()
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(blob)
        detections = detector.forward()

        if len(detections) > 0:
            for i in range(0, detections.shape[2]):
                try:
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.6:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        cv2.putText(frame_copy, 'Face Detected', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                        cv2.rectangle(frame_copy, (startX, startY), (endX, endY), (0, 0, 255), 2)
                        face = frame[startY: endY, startX: endX]
                        face_expression = cv2.resize(face, (96, 96))
                        face_expression = face_expression.reshape(1, 96, 96, 3)
                        face_expression = face_expression / 255.0
                        pred = model.predict(face_expression)
                        pred = pred.argmax()
                        if pred == 0:
                            text = 'Happy'
                        else:
                            text = 'Sad'
                        cv2.putText(frame_copy, text, (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                except Exception as e:
                    print(str(e))
                    continue
        cv2.imshow('frame', frame_copy)
		if cv2.waitKey(1) & 0xff == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
	detect_expression(model)