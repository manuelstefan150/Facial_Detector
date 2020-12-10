import cv2

# Detects faces and smiles
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')

# Gets camera feed
camera = cv2.VideoCapture(0)

# Shows current frame
while True:

    # Reads current frame from camera
    successful_frame_read, frame = camera.read() # will run forever under while true

    # Abort if error is present
    if not successful_frame_read:
        break

    # Converts to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detects faces
    faces = face_detector.detectMultiScale(frame_grayscale)
    
    # Runs face detection within each of the faces
    for (x, y, w, h) in faces:

        # Draws rectangle around faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4) # gbr, thickness (for numbers)
    
        # Gets subframe with numpy N-dimensional array slicing
        the_face = frame[y:y+h, x:x+w]

        # Converts to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)

        eyes = eye_detector.detectMultiScale(face_grayscale, scaleFactor=1.3, minNeighbors=10)

        # Finds all smiles in the face
        for (x_, y_, w_, h_) in smiles:

            # Draws a rectangle around a smile
            cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_), (50, 50, 200), 4) # gbr, thickness (for numbers)

        # Finds all eyes in the face
        for (x_, y_, w_, h_) in eyes:

            # Draws a rectangle around eye
            cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_), (255, 255, 255), 4) # gbr, thickness (for numbers)

        # Labels a smiling face
        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

        # Labels the eye(s)
        if len(eyes) > 0:
            cv2.putText(frame, 'Eyes', (x, y+h+90), manuefontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    # Shows current frame
    cv2.imshow('Smiling', frame)

    # Display
    cv2.waitKey(1)

# Cleanup
camera.release()
cv2.destroyAllWindows()