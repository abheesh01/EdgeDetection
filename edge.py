# Project written in Python which uses the Open Cv2 Library to display grainy background when user's camera is prompted 
# and face is recognized 

import cv2

# Haar Cascades face detection
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
video_capture = cv2.VideoCapture(0)

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    # Adjust parameters for fine-grained detection
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.02, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

while True:
    result, video_frame = video_capture.read()
    if result is False:
        break

    faces = detect_bounding_box(video_frame)

    cv2.imshow("My Fine-Grained Face Detection", video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

# Canny edge detection
def canny_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=1.0)  # Apply Gaussian blur
    # Adjust threshold values for fine-grained edge detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    return blurred, edges

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Image not captured')
            break

        blurred, edges = canny_edge_detection(frame)

        cv2.imshow("Fine-Grained Edge Detection", edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()