import cv2
import pandas as pd 
from datetime import datetime  # Import datetime module
from detection_of_image import detection 

# Encodings of the known images 
det = detection()
det.load_encoding_images("knownimages")

camera = cv2.VideoCapture(0)

# Create a DataFrame to store detected faces and timestamps
detected_faces = pd.DataFrame(columns=["Name", "Timestamp"])

while True:
    ret, frame = camera.read()

    # Detect the faces
    face_locations, face_names = det.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        # Add detected name and timestamp to the DataFrame if it's not already present
        if name not in detected_faces["Name"].values:
            # Get the current date and time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Create a new DataFrame for the new entry
            new_entry = pd.DataFrame({"Name": [name], "Timestamp": [current_time]})
            # Concatenate the new entry with the existing DataFrame
            detected_faces = pd.concat([detected_faces, new_entry], ignore_index=True)

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()

# Write the DataFrame to an Excel file
detected_faces.to_excel("detected_faces.xlsx", index=False)