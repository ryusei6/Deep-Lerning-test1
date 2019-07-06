import cv2

face_cascade_path = 'haarcascades/haarcascade_frontalface_default.xml'
eye_cascade_path = 'haarcascades/haarcascade_eye.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

src = cv2.imread('lena.jpg')
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(src_gray)

# 矩形
def rectangle():
    for x, y, w, h in faces:
        cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = src[y: y + h, x: x + w]
        face_gray = src_gray[y: y + h, x: x + w]
        eyes = eye_cascade.detectMultiScale(face_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

# モザイク
def mosaic():
    ratio = 0.05
    for x, y, w, h in faces:
        small = cv2.resize(src[y: y + h, x: x + w], None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
        src[y: y + h, x: x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def main():
    # rectangle()
    mosaic()
    cv2.imwrite('processed.jpg', src)

if __name__ == '__main__':
    main()
