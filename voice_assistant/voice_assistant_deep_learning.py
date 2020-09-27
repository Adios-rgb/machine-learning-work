import pyttsx3
from datetime import datetime
import speech_recognition as sr
import sounddevice as sd
from scipy.io.wavfile import write
import os
import numpy as np
import librosa
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.optimizers import Adam
from keras.utils import np_utils
import wikipedia
import webbrowser
import multiprocessing, playsound
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import ElementNotVisibleException, ElementNotInteractableException
from mutagen.mp3 import MP3
import time
import random


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

# This method speaks the input taken from user
def speak(text):
    engine.say(text)
    engine.runAndWait()
	

# This method greets user according to time
def greet():
    hour = int(datetime.now().hour)
    if hour > 0 and hour < 12:
        speak('Good Morning! this is Betty at your service')
    elif hour >= 12 and hour < 18:
        speak('Good Afternoon! this is Betty at your service')
    else:
        speak('Good Evening! this is Betty at your service')
    speak('How may I assist you?')
	

# This method takes input/command from the user	
def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as src:
        r.adjust_for_ambient_noise(src)
        print('Listening..')
        audio = r.listen(src, timeout=2, phrase_time_limit=10)
    try:
        print('Recognizing..')
        said = r.recognize_google(audio, language='en-in')
        print(f'You said: {said}\n')
    except Exception as e:
        print(e)
        print('Say that again please..')
        said = 'None'
    return said


# This method authenticates whether the user is known or unknown
def authenticate(path_to_voice_model, path_to_voice_folder):
    model = load_model(path_to_voice_model)
    take_voice_sample(path_to_voice_folder)
    try:
        files = glob.glob('{}/{}'.format(path_to_voice_folder, '*.wav'))
        files.sort(key=os.path.getmtime, reverse=True)
        file = files[0]
        print('f ', file)
        X, sample_rate = librosa.load(file, res_type='kaiser_fast') 
          # we extract mfcc feature from data
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    except Exception:
        print("Error encountered while parsing file: ", file)
        return None, None
    feature = mfccs
    feature = feature.reshape(1, -1, 1)
    pred = model.predict_classes(feature)
    return pred[0]
	

# This method takes voice sample form user and saves it to given path as audio .wav file
def take_voice_sample(path_to_voices_folder):
    files = os.listdir(path_to_voices_folder)
    latest_file_idx = 0
    if len(files) == 0:
        latest_file_idx = 1
    else:
        lst = sorted([int(y.split('_')[1]) for y in [x.split('.')[0] for x in files]], reverse=True)
        latest_file_idx = lst[0] + 1
    fs = 44100  # Sample rate
    seconds = 4  # Duration of recording
    time.sleep(1)
    speak('Please speak')
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    speak('Let me see if I know you')
    write('{}/{}_{}.wav'.format(path_to_voices_folder, 'audio', latest_file_idx), fs, myrecording)


'''This method should be used in order to record multiple voice samples from user
   the voice recognition model will be trained on this dataset'''

def record_voice_samples(path_to_sample_voices_folder):
    files = os.listdir(path_to_sample_voices_folder)
    latest_file_idx = 0
    if len(files) == 0:
        latest_file_idx = 1
    else:
        lst = sorted([int(y.split('_')[1]) for y in [x.split('.')[0] for x in files]], reverse=True)
        latest_file_idx = lst[0] + 1
    fs = 44100  # Sample rate
    seconds = 4  # Duration of recording
    time.sleep(1)
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('{}/{}_{}.wav'.format(path_to_sample_voices_folder, 'sample', latest_file_idx), fs, myrecording)


'''This method is used to parse the voice samples collected and converts audio files
   into deep learning trainable dataset and exports it as a csv file to given path'''
# path: path to audio samples folder
def parser(path):
   # function to load files and extract features
    file_name = os.listdir(path)
    l = len(file_name)
    lst = []
    for file in file_name:
   # handle exception to check if there isn't a file which is corrupted
        try:
          # here kaiser_fast is a technique used for faster extraction
            X, sample_rate = librosa.load(os.path.join(path, file), res_type='kaiser_fast') 
          # we extract mfcc feature from data
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
        except Exception as e:
            print("Error encountered while parsing file: ", file)
            return None, None

        feature = mfccs
        without_extra_slash = os.path.normpath(path)
        last_part = os.path.basename(without_extra_slash)
        label = last_part
        lst.append([feature, label])
    lst = np.array(lst)
    lst = lst.reshape(l, -1)
    data = pd.DataFrame(lst)
    data.to_csv('path to voice features csv file features.csv')


# This method further processes the csv data to preprocess it in order to be passed to our deep learning model
def data_processing_for_model(path_to_csv):
	data = pd.read_csv(path_to_csv)
	X = data.iloc[:, :-1]
	X = np.array(X)
	X = np.expand_dims(X, axis=0)
	rows = X.shape[1]
	cols = X.shape[2]
	X = X.reshape(rows, cols, 1)
	y = np.array(data.iloc[:, -1])
	lb = LabelEncoder()
	y = np_utils.to_categorical(lb.fit_transform(y))
	return X, y


# Simple deep learning model for voice recognition, can be altered depending on your dataset
def train_voice_model(path_to_save_voice_model):
	X, y = data_processing_for_model(path_to_csv)
	num_labels = y.shape[1]
	model = Sequential()
	model.add(Convolution1D(64, 2, input_shape=(40,1)))
	model.add(Activation('relu'))
	model.add(Convolution1D(64, 2))
	model.add(MaxPooling1D(2))
	model.add(Activation('relu'))
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.3))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
	model.fit(X, y, epochs=20)
	model.save('path_to_save_voice_model voice_model.h5')
	return model


'''This method is used to launch chrome and open youtube for music buffering
   customized to play either playlist or single tracks using playlist_renderer parameter'''
def launch_browser_open_site(music_query, playlist_renderer=True):
	# chromedriver needs to be downloaded
    driver = webdriver.Chrome('path to chromedriver.exe')
    driver.get("https://youtube.com")
    placeholder = WebDriverWait(driver, 10).until(ec.visibility_of_element_located((By.XPATH, "//input[@id='search']")))
    placeholder.send_keys(music_query)
    tag = 'ytd-playlist-renderer' if playlist_renderer else 'ytd-video-renderer'
    thumbnail = 'ytd-playlist-thumbnail' if tag == 'ytd-playlist-renderer' else 'ytd-thumbnail'
    WebDriverWait(driver, 10).until(ec.visibility_of_element_located((By.XPATH, "//button[@id='search-icon-legacy']"))).click()
    for _ in range(5):
        try:
            time.sleep(2)
            parentElement = driver.find_element_by_class_name("style-scope ytd-section-list-renderer")
            elementList = parentElement.find_elements_by_tag_name(tag)
            num = random.randint(1, len(elementList))
            if tag == 'ytd-video-renderer':
                WebDriverWait(driver, 10).until(ec.visibility_of_element_located((By.XPATH, f"//div[@id='primary' and @class='style-scope ytd-two-column-search-results-renderer']/ytd-section-list-renderer/div[2]/ytd-item-section-renderer/div[3]/{tag}[{num}]/div/{thumbnail}"))).click()
            else:
                WebDriverWait(driver, 10).until(ec.visibility_of_element_located((By.XPATH, f"//div[@id='primary' and @class='style-scope ytd-two-column-search-results-renderer']/ytd-section-list-renderer/div[2]/ytd-item-section-renderer/div[3]/{tag}[{num}]/{thumbnail}"))).click()
        except Exception as e:
            pass

			

protoPath = "path to deploy.prototxt.txt"
modelPath = "path to res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch('path to openface.nn4.small2.v1.t7')
model = keras.models.load_model('path to trained expression detection model model.h5')
def detect_expression_and_age(model):
    happy_count = 0
    age_dict = {"(0-2)": 0, "(4-6)": 0, "(8-12)": 0, "(15-20)": 0, "(25-32)": 0, "(38-43)": 0, "(48-53)": 0, "(60-100)": 0}
    decade_dict = {"(0-2)": '2020', "(4-6)": '2020', "(8-12)": '2015', "(15-20)": '2010-2020', "(25-32)": '2000-2020', "(38-43)": '1990', "(48-53)": '1980', "(60-100)": '1970'}
    sad_count = 0
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
                        box = detections[0, 0, i, 3: 7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        cv2.putText(frame_copy, 'Face Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                        cv2.rectangle(frame_copy, (startX, startY), (endX, endY), (0, 0, 255), 2)
                        face = frame[startY: endY, startX: endX]
                        face_expression = cv2.resize(face, (96, 96))
                        face_expression = face_expression.reshape(1, 96, 96, 3)
                        face_expression = face_expression / 255.0
                        pred = model.predict(face_expression)
                        face = cv2.resize(frame, (227, 227))
                        age = detect_age(face)[0]
                        age_dict[age] += 1
                        pred = pred.argmax()
                        if pred == 0:
                            happy_count += 1
                            text = f'Mood: Happy   Age Bracket: {age}'
                        else:
                            sad_count += 1
                            text = f'Mood: Sad   Age Bracket: {age}'
                        cv2.putText(frame_copy, text, (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
                except Exception as e:
                    print(str(e))
                    continue
        cv2.imshow('frame', frame_copy)
        cv2.waitKey(1)
        if happy_count + sad_count == 51:
            break
    cap.release()
    cv2.destroyAllWindows()
    age_actual = max(age_dict, key=age_dict.get)
    year = decade_dict[age_actual]
    if happy_count > sad_count:
        return 1, age_actual, year
    else:
        return 2, age_actual, year


def detect_age(path_to_img):
	try:
		Age_buckets = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
		# Download the age_deploy.prototxt and age_net.caffemodel files
		prototxt_path = 'path to age_deploy.prototxt'
		weights_path = 'path to age_net.caffemodel'
		ageNet = cv2.dnn.readNet(prototxt_path, weights_path)
		blob = cv2.dnn.blobFromImage(img, 1.0, (227,227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
		ageNet.setInput(blob)
		preds = ageNet.forward()
		i = preds[0].argmax()
		age = Age_buckets[i]
		ageConfidence = preds[0][i]
		return age, ageConfidence
    except Exception as e:
        print(str(e))
        pass

		
if __name__ == '__main__':
    count = 1
	authentication_required = False
	if authentication_required:
		start = False
		auth = authenticate('path to trained voice recognition model voice_model.h5', 'path to folder where voice is recorded')
		# Here 0 is the class predicted for my voice, feel free to change it accordingly
		if auth == 0:
			start = True
			speak('Hello Sir, how are you doing today?')
		else:
			speak('Your voice is not recognized')
	else:
		start = True
    while start:
        try:
            if count == 1:
                greet()
                count = 2
            query = takeCommand().lower()
            if 'search' in query or 'wiki' in query or 'wikipedia' in query:
                speak('Searching Wikipedia..')
                results = wikipedia.summary(query, sentences=3)
                speak('according to wikipedia')
                speak(results)
				
            elif 'youtube' in query or 'play music' in query:
                speak('what type of music would you like to listen to?')
                music_query = takeCommand().lower()
				# If you want to listen to a specific song by specific artist
                if 'play' in music_query and 'by' in music_query:
                    music_query = music_query[music_query.index('play') + 5:]
                    driver = webdriver.Chrome('path to chromedriver.exe')
                    driver.get("https://youtube.com")
                    placeholder = WebDriverWait(driver, 10).until(ec.visibility_of_element_located((By.XPATH, "//input[@id='search']")))
                    placeholder.send_keys(music_query)
                    WebDriverWait(driver, 10).until(ec.visibility_of_element_located((By.XPATH, "//button[@id='search-icon-legacy']"))).click()
                    for _ in range(5):
                        try:
                            time.sleep(1)
                            WebDriverWait(driver, 10).until(ec.visibility_of_element_located((By.XPATH, "//div[@id='contents']/ytd-item-section-renderer/div[3]/ytd-video-renderer[1]/div/ytd-thumbnail"))).click()
                        except Exception:
                            pass
				# If you want to play something non specific e.g. play Rock, play hip hop, play blues etc.
                else:
                    music_query = music_query[music_query.index('play') + 5:] + ' playlist'
                    launch_browser_open_site(music_query, True)
					

            elif 'play offline' in query or 'play from storage' in query:
                sngs_lst = os.listdir('path to songs')
                random.shuffle(sngs_lst)
                for i in sngs_lst[:2]:
                    time.sleep(2)
                    t1 = time.time()
                    t2 = t1 + 1
                    audio = MP3(os.path.join('path to songs', i))
                    p = multiprocessing.Process(target=playsound.playsound, args=(os.path.join('path to songs', i),))
                    p.start()
                    while (t2 - t1) < audio.info.length:
                        sub_quer = takeCommand().lower()
                        print(sub_quer)
                        t2 = time.time()
                        if 'cut' in sub_quer or 'stop' in sub_quer:
                            p.terminate()
                            print('terminated')
                            break
                    break
					
					
            elif 'temperature' in query or 'weather' in query:
                driver = webdriver.Chrome('path to chromedriver.exe')
                driver.get("https://google.com")
                driver.minimize_window()
                placeholder = WebDriverWait(driver, 10).until(ec.visibility_of_element_located((By.XPATH, "//input[@title='Search']")))
                placeholder.send_keys(query)
                time.sleep(1)
                WebDriverWait(driver, 10).until(ec.visibility_of_element_located((By.XPATH, "//input[@title='Search']"))).send_keys(Keys.RETURN)
                temp = WebDriverWait(driver, 10).until(ec.visibility_of_element_located((By.XPATH, "//div[@class='vk_bk sol-tmp']/span[@id='wob_tm']"))).text
                precip = WebDriverWait(driver, 10).until(ec.visibility_of_element_located((By.XPATH, "//div[@class='vk_gy vk_sh wob-dtl']/div/span"))).text
                speak(f'According to AccuWeather, temperature for today will be {temp} degree celcius, with {precip} chance of precipitation')
                driver.close()
				
				
            elif 'play' in query and 'mood' in query:
                speak('Yes surely, let me see if I can detect your mood, and I will play some songs accordingly. Please look directly at the camera.')
                pred, _, year = detect_expression_and_age(model)
                if pred == 2:
                    music_query = f'{year} best motivational playlist'
                    speak("Sir, you don't look happy to me. Let me lift your spirits")
                    launch_browser_open_site(music_query, True)
                else:
                    music_query = f'{year} best songs playlist'
                    speak('Sir, you look happy, here is what I think will suit your mood')
                    launch_browser_open_site(music_query, True)

            elif 'close' in query or 'terminate' in query or 'turn off' in query:
                speak('Thanks for having me at your service, have a good day')
                start = False

            else:
                count += 1
                if count % 5 == 0:
                    speak('My responses are limited, you must speak the right keywords')

        except ElementNotInteractableException:
            pass
        except Exception as e:
            print(e)
