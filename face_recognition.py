import face_recognition  # For face detection and recognition
import cv2  # For image and video processing

# Load an image file into a numpy array
# Load an image file of a person for face recognition
image_of_person = face_recognition.load_image_file("person.jpg")
# Encode the face in the image
person_face_encoding = face_recognition.face_encodings(image_of_person)[0]

# List of known face encodings for comparison
known_face_encodings = [person_face_encoding]
# Corresponding names for the known faces
known_face_names = ["Person's Name"]

# Start video capture from the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a single frame from the video
    ret, frame = video_capture.read()

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame)

    # Iterate through detected face locations and their corresponding encodings
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face encoding with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"
        # If a match is found, use the name of the known face
        if True in matches:
            # Get the index of the first matching face
            first_match_index = matches.index(True)
            # Retrieve the name corresponding to the matched face
            name = known_face_names[first_match_index]

        # Draw a rectangle around the face and label it
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
       # Put the name of the person above the rectangle
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object to free up resources
video_capture.release()
# Close all OpenCV windows
cv2.destroyAllWindows()