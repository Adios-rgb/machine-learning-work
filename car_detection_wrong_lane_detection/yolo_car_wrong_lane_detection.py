import cv2
import numpy as np

MIN_CONF=0.5

def detect_object(frame, net, ln, indx=0):
    (H, W) = frame.shape[:2]
    results = []
    # construct a blob, perform a forward pass using YOLO which will return bounding boxes and probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    centroids = []
    confidences = []
    for output in layerOutputs:
    # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # check if probability is greater than the minimum probability and classID is same as of Car
            if confidence > MIN_CONF and classID == indx:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)
    return results
	
	
if __name__ == '__main__':
	labelsPath = "coco.names"
	LABELS = open(labelsPath).read().strip().split("\n")
	weightsPath = "yolov3.weights"
	configPath = "yolov3.cfg"
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	cap = cv2.VideoCapture('video.mp4')
	writer = cv2.VideoWriter('detection_output.avi', cv2.VideoWriter_fourcc('F','M','P','4'), 20, (854,480))
	base_violation_x = (420, 590)
	base_violation_y = 125
	count = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		results = detect_object(frame, net, ln, LABELS.index('car'))
		violate = set()
		if len(results) >= 1:
			centroids = [r[2] for r in results]
			for i, centroid in enumerate(centroids):
				if centroid[0] >= base_violation_x[0] and centroid[0] <= base_violation_x[1] and centroid[1] >= base_violation_y:
					violate.add(i)
		for (i, (prob, bbox, centroid)) in enumerate(results):
			(startX, startY, endX, endY) = bbox
			(centreX, centerY) = centroid
			color = (0,255,0)
			if i in violate:
				color = (0,0,255)
			cv2.rectangle(frame, (startX,startY), (endX,endY), color, 2)
			cv2.circle(frame, (centreX, centerY), 5, color, 1)
		text = "Wrong Way Violations: {}".format(len(violate))
		cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
		writer.write(frame)
	cap.release()
	writer.release()