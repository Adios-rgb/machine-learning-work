import PySimpleGUI as sg
import os.path
import numpy as np
import cv2, sys, imutils, glob
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


layout = [
      [sg.Image(filename='', key='image')],
      [sg.Text('Enter Path to image file'), sg.InputText(key='folder'), sg.Button("OK"), sg.Button('Previous'), sg.Button('Next')],
      [sg.Radio('threshold', 'Radio', size=(10, 1), key='thresh'),
       sg.Slider((0, 255), 128, 1, orientation='h', size=(30, 12), key='thresh_slider_min'),
       sg.Slider((0, 255), 128, 1, orientation='h', size=(30, 12), key='thresh_slider_max')],
      [sg.Radio('invert', 'Radio', size=(10, 1), key='invert')],
      [sg.Radio('canny', 'Radio', size=(10, 1), key='canny'),
       sg.Slider((0, 255), 128, 1, orientation='h', size=(30, 12), key='canny_slider_a'),
       sg.Slider((0, 255), 128, 1, orientation='h', size=(30, 12), key='canny_slider_b')],
      [sg.Radio("erode", "Radio", size=(10, 1), key="erode"),
       sg.Slider((1, 5), 1, 1, orientation="h", size=(30, 12), key="erode_kernel_slider"),
       sg.Slider((1, 5), 1, 1, orientation="h", size=(30, 12), key="eriterations_slider")],        
      [sg.Radio("dilate", "Radio", size=(10, 1), key="dilate"),
       sg.Slider((1, 5), 1, 1, orientation="h", size=(30, 12), key="dilate_slider"),
       sg.Slider((1, 5), 1, 1, orientation="h", size=(30, 12), key="diliterations_slider")],        
      [sg.Radio("opening", "Radio", size=(10, 1), key="open"),
       sg.Slider((1, 5), 1, 1, orientation="h", size=(30, 12), key="opening_slider")],
      [sg.Radio("closing", "Radio", size=(10, 1), key="close"),
       sg.Slider((1, 5), 1, 1, orientation="h", size=(30, 12), key="closing_slider")],
      [sg.Radio('all contours', 'Radio', size=(10, 1), key='contour')],
      [sg.Radio('convex hull', 'Radio', size=(10, 1), key='convex_hull')],
      [sg.Radio('find shapes', 'Radio', key='shapes')],
      [sg.Radio('find contours with aspect ratio', 'Radio', key='contours_aspect'),
       sg.Slider(change_submits=True, range=(0, 15), size=(30, 12), resolution=0.1, orientation='h', default_value=1, key='contours_ar_slider_min'),
       sg.Slider(change_submits=True, range=(0, 15), size=(30, 12), resolution=0.1, orientation='h', default_value=1, key='contours_ar_slider_max')],
      [sg.Radio('blur', 'Radio', size=(10, 1), key='blur'),
       sg.Slider((1, 11), 1, 1, orientation='h', size=(30, 12), key='blur_slider')],
      [sg.Radio('masking hsv', 'Radio', size=(10, 1), key='masking'),
       sg.Slider((0, 255), 0, 1, orientation='h', size=(18, 10), key='h_slider_l'),
       sg.Slider((0, 255), 0, 1, orientation='h', size=(18, 10), key='s_slider_l'),
       sg.Slider((0, 255), 0, 1, orientation='h', size=(18, 10), key='v_slider_l'),
       sg.Slider((0, 255), 255, 1, orientation='h', size=(18, 10), key='h_slider_u'),
       sg.Slider((0, 255), 255, 1, orientation='h', size=(18, 10), key='s_slider_u'),
       sg.Slider((0, 255), 255, 1, orientation='h', size=(18, 10), key='v_slider_u')],
      [sg.Radio('enhance', 'Radio', size=(10, 1), key='enhance'),
       sg.Slider((1, 255), 128, 1, orientation='h', size=(20, 12), key='enhance_slider')],
      [sg.Text('Zoom'), sg.InputCombo(values=[1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4], default_value=1, size=(10, 1), enable_events=True, key='input_combo'),
      sg.Text('Psm'), sg.InputCombo(values=[3, 6, 7, 9, 11, 12, 13], default_value=3, size=(10, 1), enable_events=True, key='input_psm')],
      [sg.Button('Save', size=(10, 1)), sg.Button('Get Text', size=(10, 1)), sg.Button('Reset', size=(10, 1)), sg.Button('Exit', size=(10, 1))]
    ]

window = sg.Window('analysis', resizable=True).Layout([
                            [sg.Column(layout, size=(1900, 1500), scrollable=True, key = "Column")]])

global save, save_path, reset, img_path, img_valid
save, reset, img_path, img_valid, previous, counter = False, False, os.getcwd(), False, False, 1
save_path = os.path.join(os.getcwd(), 'temp_files')
while True:
    try:
        if not save and not reset:
            original = img_path
            if original.endswith('.png') or original.endswith('.jpg'):
                frame = cv2.imread(original)
                img_valid = True
        elif save and not reset and not previous:
            list_of_files = glob.glob(save_path + '/*.png')
            latest_file = max(list_of_files, key=os.path.getctime)
            frame = cv2.imread(latest_file)
        elif save and not reset and previous:
            list_of_files = glob.glob(save_path + '/*.png')
            list_of_files = sorted(list_of_files, key=os.path.getctime)
            total_files = len(list_of_files)
            frame = cv2.imread(list_of_files[total_files - counter])
        elif not save and reset:
            frame = cv2.imread(original)
        
        event, values = window.read(timeout=0, timeout_key='timeout')
        
        if event == 'OK':
            path = values['folder']
            if os.path.exists(path):
                img_path = path
            else:
                sg.PopupCancel('Path or image does not exists')
                
        if event == 'Previous':
            previous = True
            if counter < 5:
                counter += 1
                for key, val in values.items():
                    if val is True:
                        window.FindElement(key).Update(False)
            
        if event == 'Next':
            print('counter here', counter)
            if previous and counter != 1:
                counter -= 1
                for key, val in values.items():
                    if val is True:
                        window.FindElement(key).Update(False)

        if event == 'Exit' or event == sg.WIN_CLOSED:
            fileslist = glob.glob(save_path + '/*.png')
            if len(fileslist) > 0:
                choice = sg.PopupOKCancel("Do you want to delete all the metadata as well?")
                if choice == 'OK':
                    [os.remove(os.path.join(save_path, f)) for f in fileslist]
                    break
            break

        if values['invert']:
            if len(frame.shape) > 2:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.bitwise_not(gray)
            
        if values['thresh']:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.threshold(frame, values['thresh_slider_min'], values['thresh_slider_max'], cv2.THRESH_BINARY)[1]

        if values['canny']:
            frame = cv2.Canny(frame, values['canny_slider_a'], values['canny_slider_b'])

        if values["erode"]:
            kernel_erode = np.ones((int(values["erode_kernel_slider"]), int(values["erode_kernel_slider"])), dtype=np.uint8)
            frame = cv2.erode(frame, kernel_erode, iterations=int(values["eriterations_slider"]))

        if values["dilate"]:
            kernel_dil = np.ones((int(values["dilate_slider"]), int(values["dilate_slider"])), dtype=np.uint8)
            frame = cv2.dilate(frame, kernel_dil, iterations=int(values["diliterations_slider"]))

        if values["open"]:
            kernel_open = np.ones((int(values["opening_slider"]), int(values["opening_slider"])), dtype=np.uint8)
            frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel_open)

        if values["close"]:
            kernel_close = np.ones((int(values["closing_slider"]), int(values["closing_slider"])), dtype=np.uint8)
            frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel_close)
            
        if values['contour']:
            if len(frame.shape) > 2:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cv2.drawContours(frame, cnts, -1, (0, 0, 255), 2)
            
        if values['convex_hull']:
            if len(frame.shape) > 2:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            for cnt in cnts:
                hull = cv2.convexHull(cnt, returnPoints=True)
                cv2.drawContours(frame, [hull], -1, (0, 0, 255), 2)
            
        if values['shapes']:
            if len(frame.shape) > 2:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            for cnt in cnts:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) == 3:
                    shape = 'triangle'
                elif len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    ar = w / h
                    shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
                elif len(approx) == 5:
                    shape = "pentagon"
                elif len(approx) == 6:
                    shape = "hexagon"
                elif len(approx) == 7:
                    shape = "heptagon"
                elif len(approx) == 8:
                    shape = "octagon"
                elif len(approx) == 9:
                    shape = "nonagon"
                else:
                    shape = 'circle'
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
                    cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    
        if values['contours_aspect']:
            if len(frame.shape) > 2:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            for cnt in cnts:
                x, y, w, h = cv2.boundingRect(cnt)
                if w / h >= values['contours_ar_slider_min'] and w / h <= values['contours_ar_slider_max']:
                    cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)

        if values['blur']:
            frame = cv2.GaussianBlur(frame, (21, 21), values['blur_slider'])

        if values['masking']:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([values['h_slider_l'], values['s_slider_l'], values['v_slider_l']]), np.array([values['h_slider_u'], values['s_slider_u'], values['v_slider_u']]))
            frame = mask

        if values['enhance']:
            enh_val = values['enhance_slider'] / 40
            clahe = cv2.createCLAHE(clipLimit=enh_val, tileGridSize=(8, 8))
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        if event == 'Save':
            save = True
            reset = False
            previous = False
            counter = 1
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            list_of_files = glob.glob(save_path + '/*.png') # * means all if need specific format then *.csv
            if len(list_of_files) == 0:
                file_name = 'img_1'
            else:
                latest_file = max(list_of_files, key=os.path.getctime)
                file_index = int(latest_file.split('.')[0].split('img_')[1]) + 1
                file_name = 'img_' + str(file_index)
            cv2.imwrite(os.path.join(save_path, file_name) + '.png', frame)

        if event == 'Reset':
            save = False
            reset = True
            window.FindElement('thresh_slider_min').Update(128)
            window.FindElement('thresh_slider_max').Update(128)
            window.FindElement('canny_slider_a').Update(128)
            window.FindElement('canny_slider_b').Update(128)
            window.FindElement('erode_kernel_slider').Update(1)
            window.FindElement('eriterations_slider').Update(1)
            window.FindElement('dilate_slider').Update(1)
            window.FindElement('diliterations_slider').Update(1)
            window.FindElement('opening_slider').Update(1)
            window.FindElement('closing_slider').Update(1)
            window.FindElement('blur_slider').Update(1)
            window.FindElement('h_slider_l').Update(0)
            window.FindElement('s_slider_l').Update(0)
            window.FindElement('v_slider_l').Update(0)
            window.FindElement('h_slider_u').Update(255)
            window.FindElement('s_slider_u').Update(255)
            window.FindElement('v_slider_u').Update(255)
            window.FindElement('enhance_slider').Update(128)
            window.FindElement('input_combo').Update(1)
            window.FindElement('input_psm').Update(3)
            window.FindElement('contours_ar_slider_min').Update(1)
            window.FindElement('contours_ar_slider_max').Update(1)
            for key, val in values.items():
                if val is True:
                    window.FindElement(key).Update(False)

        if event == 'Get Text':
            if values['input_combo']:
                frame = cv2.resize(frame, (int(frame.shape[1] * values['input_combo']), int(frame.shape[0] * values['input_combo'])))
            txt = pytesseract.image_to_string(frame, config='--psm {} --oem 3'.format(values['input_psm']))

            layout_text = [[sg.Text('Extracted text')], [sg.Multiline(txt, key="multiline_text", disabled=True, autoscroll=True)]]
            window_text = sg.Window("Text Window", layout_text, modal=True)
            while True:
                event, values = window_text.read()
                if event == "Exit" or event == sg.WIN_CLOSED:
                    break
            window_text.close()
        
        if img_valid:
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            window['image'].update(data=imgbytes)
        
    except Exception as e:
        print(e)
        exception_type, exception_object, exception_traceback = sys.exc_info()
        line_number = exception_traceback.tb_lineno
        print(line_number)
        break

window.close()