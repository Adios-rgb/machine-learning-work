from imutils import paths
import numpy as np
import pickle
import imutils
import cv2
import os
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from four_point_transform import transform
from sklearn.preprocessing import LabelEncoder
import pickle


aug = ImageDataGenerator()
aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
total = 0
path = 'path to your dataset folders'


def generate_images(num_of_images=5):
    for label_folders in os.listdir(path):
        output_path = os.path.join(path, label_folders)
        images = os.path.join(path, label_folders)
        for img in os.listdir(images):
            image_path = os.path.join(images, img)
            image = load_img(image_path)
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            imageGen = aug.flow(image, batch_size=1, save_to_dir=output_path, 
                                save_prefix=label_folders, save_format="jpg")
            for image in imageGen:
                total += 1
                # To generate number of augmented images per image
                if total == num_of_images - 1:
                    total = 0
                    break


protoPath = 'Path to prototxt file'
modelPath = 'Path to res10_300x300_ssd_iter_140000.caffemodel file'
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch('Path to openface.nn4.small2.v1.t7 file')
imagePaths = list(paths.list_images(path))
knownEmbeddings = []
knownNames = []
detection_threshold = 0.5
min_height = 20
min_width = 20

def calc_and_pickle_embeddings():
    for (i, imagePath) in enumerate(imagePaths):
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h,w) = image.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()

        if len(detections) > 0:
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            if confidence > detection_threshold:
                box = detections[0, 0, i, 3: 7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')
                face = image[startY: endY, startX: endX]
                (fH, fW) = face.shape[: 2]
                # Sanity check for minimum height and width
                if fH < min_height or fW < min_width:
                    continue
                warped = transform.four_point_transform(image, np.array([[startX, startY], [startX + fW, startY], [startX + fW, startY + fH], [startX, startY + fH]]))
                faceBlob = cv2.dnn.blobFromImage(warped, 1.0 / 255.0, (96, 96), (0, 0, 0), swapRB=False, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1
    data = {'embeddings': knownEmbeddings, 'names': knownNames}
    f = open('path to saving the face embeddings as pickle e.g. embeddings.pickle', 'wb')
    f.write(pickle.dumps(data))
    f.close()


def pickle_model():
    data = pickle.loads(open('path to embeddings pickle file saved in previous step', 'rb').read())
    le = LabelEncoder()
    labels = le.fit_transform(data['names'])
    recognizer = RandomForestClassifier(n_estimators=200, criterion='entropy', n_jobs=-1)
    recognizer.fit(data['embeddings'], labels)

    f = open('path to dump RandomForest model as pickle', 'wb')
    f.write(pickle.dumps(recognizer))
    f.close()

    f = open('path to the dump labels_encoder as pickle', 'wb')
    f.write(pickle.dumps(le))
    f.close()


if __name__ == '__main__':
    generate_images(10)
    calc_and_pickle_embeddings()
    pickle_model()
    recognizer = pickle.loads(open('Path to load random forest pickle model', 'rb').read())
    le = pickle.loads(open('Path to load labels_encoder pickle file', 'rb').read())

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > detection_threshold:
                box = detections[0, 0, i, 3: 7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')
                face = frame[startY: endY, startX: endX]
                (fH, fW) = face.shape[: 2]
                if fW < min_width or fH < min_height:
                    continue
                warped = transform.four_point_transform(frame, np.array([[startX, startY], [startX + fW, startY], [startX + fW, startY + fH], [startX, startY + fH]]))
                faceBlob = cv2.dnn.blobFromImage(warped, 1.0 / 255.0, (96, 96), (0, 0, 0), swapRB=False, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()