import cv2
import requests
import numpy as np

# URL of the PHP server
php_server_url = 'http://streaming.hibowl.com/php_server/video_stream.php'  # Replace with the actual URL

# Video capture from webcam (adjust the index if using a different video source)
cap = cv2.VideoCapture("output/output1.avi")

print("ok1")

count = 0

while True:
    
    # Read a frame from the video capture
    success, frame = cap.read()

    if not success:
        count+=1
        continue

    elif count%10 == 0:
        print("he")

        # Convert the frame to JPEG format
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        print("he1")
        # Send the JPEG image to the PHP server
        try:
            print("he2")
            response = requests.post(php_server_url, data=img_bytes, headers={'Content-Type': 'image/jpeg'})
            print(response)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error sending image: {e}")

        print("he3")
        # Display the frame locally (optional)
        #cv2.imshow('Local Display', frame)
        #cv2.imwrite("theimage.jpg", img_encoded)

    count+=1


    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
