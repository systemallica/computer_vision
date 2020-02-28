import cv2

# Read video file
video = cv2.VideoCapture('video1.mp4')

# Check if camera opened successfully
if not video.isOpened():
    print("Error opening video stream or file")

# Read until video is completed
while video.isOpened():
    # Capture frame-by-frame
    ret, frame = video.read()

    if ret:
        # Display the resulting frame
        resizedFrame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("frame", resizedFrame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
video.release()

# Closes all the frames
cv2.destroyAllWindows()