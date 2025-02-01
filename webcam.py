import cv2

# Set the camera ID (usually 1 for an external camera)
camera_id = 0

# Open the camera
cap = cv2.VideoCapture(camera_id)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))  # Detect faces
    
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)  # Draw bounding box

    return faces

# Read frames in a loop
while True:
    ret, frame = cap.read()  # Read frame from the camera
    if not ret:
        print("Failed to grab frame")   
        break

    detect_bounding_box(frame)  # Detect and draw bounding boxes

    # Display the frame
    cv2.imshow("External Camera Feed", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
