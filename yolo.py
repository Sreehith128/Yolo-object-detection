import cv2
import numpy as np 

#This function is used to preprocess image and create
# a deep neural network using pretrained weights and
# cfg file which contains architecture
def yolov3(Img):
	#download weights and cfg file from pjreddie.com
	yoloNet = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
	classes=list()
	#extracting class names from txt file(from pjreddie.com)
	with open('yolov3.txt') as f:
		classes=[i.rstrip() for i in f.readlines()]

	layer_names=yoloNet.getLayerNames()
	#we get all yolo layers in out_layers
	out_layers=[layer_names[i[0]-1] for i in yoloNet.getUnconnectedOutLayers()]
	#colors for different classes
	colors=np.random.uniform(0,255,size=(len(classes),3))

	img=cv2.resize(Img,None,fx=0.4,fy=0.4)
	height,width,_ = img.shape
	#to retain original height and width after resizing
	height=height*2.5
	width=width*2.5
	#converting image to blob(Binary Large Object)
	#blob is used to extract features for neural network
	blob =cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),True,crop=False)
	yoloNet.setInput(blob)
	outs=yoloNet.forward(out_layers)
	return outs,width,height,colors,classes

#calculations for getting box cordinates and confidences of detected boxes
def calculations(outs,width,height):
	class_idx=list()
	confidences=list()
	boxes=list()

	for out in outs:
	    for i in out:
	    	#confidences are from fifth index
	        scores = i[5:]
	        class_id = np.argmax(scores)
	        confidence = scores[class_id]
	        #0.5 is threshold confidence 
	        if confidence > 0.5:
	            # first two indices of out are centre co-ordinates of box
	            center_x = int(i[0] * width)
	            center_y = int(i[1] * height)
	            #second two indices are width and height resp.
	            w = int(i[2] * width)
	            h = int(i[3] * height)
	            # x,y are first corner co-ordinates 
	            x = int(center_x - w / 2)
	            y = int(center_y - h / 2)
	            boxes.append([x, y, w, h])
	            confidences.append(float(confidence))
	            class_idx.append(class_id)
	#applying non-max supression to remove excess boxes on an object
	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
	return boxes,indexes,confidences,class_idx

#drawing yolo boxes on image
def YoloShow(boxes,indexes,confidences,class_idx,colors,classes,img):
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
	    if i in indexes:
	        x, y, w, h = boxes[i]
	        label = str(classes[class_idx[i]])
	        color = colors[i]
	        #drawing boxes using previous calculations
	        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
	        cv2.putText(img, label+'('+str(round(confidences[i],2))+')', (x, y +15), font, 2, color, 2)

	cv2.imshow("Image", img)

#main  funtion

#accessing webcam
cap=cv2.VideoCapture(0)

while True:
	_,img=cap.read()
	outs,width,height,colors,classes=yolov3(img)
	boxes,indexes,confidences,class_idx=calculations(outs,width,height)
	YoloShow(boxes,indexes,confidences,class_idx,colors,classes,img)
	#press 'q' to quit webcam
	#ord('q') returns ASCII of 'q'
	if cv2.waitKey(1) & 0xff == ord('q'):
		break

