import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
cap =cv2.VideoCapture(0)
while True:
    cap.set(10,160)
    ret,img = cap.read()
    img = cv2.resize(img,(480,360))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = pytesseract.image_to_string(img)
    boxes1 = pytesseract.image_to_boxes(img)
    # print(boxes1)
    # ;
    himg, wimg, z = img.shape
    boxes2 = pytesseract.image_to_data(img)
    # print(boxes2)
    for x, i in enumerate(boxes2.splitlines()):
        if x != 0:
            k = i.split()
            # print(k)
            if len(k) == 12:
                x, y, wid, hei = int(k[6]), int(k[7]), int(k[8]), int(k[9])
                cv2.rectangle(img, (x, y), (wid + x, hei + y), (0, 0, 255), 1)
                #cv2.putText(img, k[11], (x, y), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
                print(k[11])
    cv2.imshow('akhil', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


