import face_recognition
import cv2

# Load an image file into a numpy array
image_of_person = face_recognition.load_image_file("person.jpg")
# Encode the face in the image
person_face_encoding = face_recognition.face_encodings(image_of_person)[0]

# Initialize some variables
known_face_encodings = [person_face_encoding]
known_face_names = ["Person's Name"]

# Start video capture from the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a single frame from the video
    ret, frame = video_capture.read()

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face encoding with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"
        # If a match is found, use the name of the known face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle around the face and label it
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
video_capture.release()
cv2.destroyAllWindows()