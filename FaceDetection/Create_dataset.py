import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# สร้าง dataSet
def create_dataset(img,id,img_id):
    cv2.imwrite("data/pic."+str(id)+"."+str(img_id)+".jpg",img)



# แปลงภาพสีเป็น ดำ และ เรียกใช้ features เพื่อ กีเทคใบหน้า แล้ว ระบุ ตำแหน่ง x,y เพื่อวาดรูป สีเหลี่ยมพืนผ้า
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)

    coords = []

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
        coords = [x, y, w, h]
    return img, coords


# ตรวจจับใบหน้า และ เรียกใช้ draw_boundary
def detect(img, faceCascade,img_id):
    img, coords = draw_boundary(img, faceCascade, 1.1, 10, (0, 0, 255), "Face")



    # ตรวจจับเฉพาะหน้า
    if len(coords)==4 :
        id = 1
        # img(y:y+h,x:x+w)
        result =img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
        # เรียกใช้ creat_dataset
        create_dataset(result, id, img_id)

    return img

img_id = 0

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    frame = detect(frame, faceCascade,img_id)
    cv2.imshow('frame', frame)
    img_id+=1
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()