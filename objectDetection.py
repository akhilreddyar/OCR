import cv2
import numpy as np
net =  cv2.dnn.readNet('C:\\Users\priyanka\PycharmProjects\ocr2\yolov3.cfg','C:\\Users\priyanka\PycharmProjects\ocr2\yolov3.weights')
classes=[]
with open('C:\\Users\priyanka\PycharmProjects\ocr2\coco.names','r') as f:
    classes = f.read().splitlines();
print(classes)
#img = cv2.imread('C:\\Users\priyanka\PycharmProjects\ocr2\\test1ae.jpg')

cap = cv2.VideoCapture(0)
while True:
    ret,ok = cap.read()
    img = cv2.resize(ok,(480,360))
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    wid, hei, _ = img.shape
    net.setInput(blob)
    output_lay = net.getUnconnectedOutLayersNames()
    layer_out = net.forward(output_lay)
    # cv2.imshow('akh',img);
    boxes = []
    confidences = []
    class_ids = []
    for out in layer_out:
        for d in out:
            scores = d[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # print(class_id)
            # print(confidence)
            if confidence > 0.5:
                # print(class_id)
                center_x = int(d[0] * wid)
                center_y = int(d[1] * hei)
                w = int(d[2] * wid)
                h = int(d[3] * wid)
                print((center_x + int(w / 2), center_y + int(h / 2)))
                # cv2.imshow('ak',img)
                x, y, x1, y1 = center_x - int(w / 2), center_y - int(h / 2), center_x + int(w / 2), center_y + int(
                    h / 2)
                boxes.append([x, y, x1, y1])
                confidences.append(confidence)
                class_ids.append(class_id)
            # print(scores)
    for i, x in enumerate(boxes):
        cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 0, 255), 5)
        cv2.putText(img, classes[class_ids[i]],
                    (int((boxes[i][0] + boxes[i][2]) / 2), int((boxes[i][1] + boxes[i][3]) / 2)),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(img, 'confinde:' + str(confidences[i]), (boxes[i][0], boxes[i][1]), cv2.FONT_ITALIC, 1, (255, 0, 0),
                    2)
    cv2.imshow('telugu',img)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
#print(str(hei)+" "+str(wid))
cap.release()
cv2.destroyAllWindows()
