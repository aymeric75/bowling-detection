import cv2

inputVideo      = "output_pinsL3.avi"
folderOutput    = "pins_left3"

cap = cv2.VideoCapture(inputVideo)


length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("length")
print(length)


count_success = 0

ref_array = None


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:


        # chaque 50 frames
        if count_success % 10 == 0:

            print("IMG : {}".format(str(count_success)))

            # Display the annotated frame
            cv2.imwrite(folderOutput+"/imNew"+str(count_success)+".png", frame)


        count_success+=1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()





# charge la vidéo
# toutes les X frames, imgsave (ou figsave je sais plus)