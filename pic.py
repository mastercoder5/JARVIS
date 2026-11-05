import cv2

cam = cv2.VideoCapture(0)
print("SPACE = capture, ESC = cancel")
while True:
    ok, frame = cam.read()
    if not ok:
        break
    cv2.imshow("Enroll", frame)
    key = cv2.waitKey(1)
    if key == 32:
        cv2.imwrite("me.jpg", frame)
        print("Saved me.jpg")
        break
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()