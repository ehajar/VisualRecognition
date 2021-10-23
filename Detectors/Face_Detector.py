import cv2

# Load some pretrained data on face frontals from opencv (haar casrcade algo )
trained_face_data = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")

# to capture video
webcam = cv2.VideoCapture(0)
# Iterate forever over the frames
while True:
    # read the currenht frame
    successful_frame_read, frame = webcam.read()

    # Must convert  to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 250, 0), 2)
    cv2.imshow("Hajar", frame)
    key = cv2.waitKey(1)

    # Quitting using a key
    if key == 82 or key == 113:
        break
webcam.release()


# # Detect faces
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)


# # Draw rectangles around faces

# for (x, y, w, h) in face_coordinates:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 250, 0), 2)


# # cv2.imshow("Hajar", img)                       # Show image
# # cv2.waitKey(0)


# print(face_coordinates)
