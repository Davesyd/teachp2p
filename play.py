import cv2
import time
from simple_facerec import SimpleFacerec
from cvzone.HandTrackingModule import HandDetector
import cvzone
import os


# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera
cap = cv2.VideoCapture(0)

cap.set(3, 1280)

cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)

pTime = 0

cTime = 0

path = "ImagesPNG"
myList = os.listdir(path)
print(myList)


class DragImg():
    def __init__(self, path, posOrigin, imgType):

        self.posOrigin = posOrigin
        self.imgType = imgType
        self.path = path

        if self.imgType == 'png':
            self.new_method()
        else:
            self.img = cv2.imread(self.path)

        # self.img = cv2.resize(self.img, (0,0),None,0.4,0.4)

        self.size = self.img.shape[:2]

    def new_method(self):
        self.img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)

    def update(self, cursor):
        ox, oy = self.posOrigin
        h, w = self.size

        # Check if in region
        if ox < cursor[0] < ox + w and oy < cursor[1] < oy + h:
            self.posOrigin = cursor[0] - w // 2, cursor[1] - h // 2


listImg = []
for x, pathImg in enumerate(myList):
    if 'png' in pathImg:
        imgType = 'png'
    else:
        imgType = 'jpg'
    listImg.append(DragImg(f'{path}/{pathImg}', [50 + x * 300, 50], imgType))


while True:
    success, img = cap.read()

    # Drag and Drop

    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]['lmList']
        # Check if clicked
        length, info, img = detector.findDistance(lmList[8], lmList[12], img)
        print(length)
        if length < 60:
            cursor = lmList[8]
            for imgObject in listImg:
                imgObject.update(cursor)

    try:

        for imgObject in listImg:

            # Draw for JPG image
            h, w = imgObject.size
            ox, oy = imgObject.posOrigin
            if imgObject.imgType == "png":
                # Draw for PNG Images
                img = cvzone.overlayPNG(img, imgObject.img, [ox, oy])
            else:
                img[oy:oy + h, ox:ox + w] = imgObject.img

    except:
        pass

        # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(img)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        name_show = name.split("_", 1)[0]
        cv2.putText(img, name_show, (x1 + 50, y1 - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Img", img)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()

cv2.destroyAllWindows()
