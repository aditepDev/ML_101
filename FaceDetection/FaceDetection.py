import cv2


# faceCascade = cv2.CascadeClassifier('\data\haarcascade_frontalface_default.xml')
# smileCascade = cv2.CascadeClassifier('\data\haarcascade_smile.xml')

# faceCascade = cv2.CascadeClassifier('D:\top\python\transform_101\FaceDetection\data\haarcascade_frontalface_default.xml')
# smileCascade = cv2.CascadeClassifier('D:\top\python\transform_101\FaceDetection\data\haarcascade_smile.xml')

print(cv2.__file__)
faceCascade = cv2.CascadeClassifier('C:\Python37\lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('C:\Python37\lib\site-packages\cv2\data\haarcascade_smile.xml')

# ยิ้ม
def detectsmaile(img,smileCascade) :
    img,coords = draw_boundary(img,smileCascade,1.1,10,(0,0,255),"smile")
    return img

# แปลงภาพสีเป็น ดำ และ เรียกใช้ features เพื่อ กีเทคใบหน้า แล้ว ระบุ ตำแหน่ง x,y เพื่อวาดรูป สีเหลี่ยมพืนผ้า
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray,scaleFactor,minNeighbors)

    coords=[]

    for (x,y,w,h) in features :
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
        coords=[x,y,w,h]
    return img,coords

# ตรวจจับใบหน้า และ เรียกใช้ draw_boundary
def detect(img,faceCascade,smileCascade) :
    img,coords = draw_boundary(img,faceCascade,1.1,10,(0,0,255),"face")
    print(coords)
    if len(coords)==4 :
        result =img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
        img_smile = detectsmaile(result,smileCascade)
    return img


cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    frame = detect(frame,faceCascade,smileCascade)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    cv2.imshow('frame', frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
