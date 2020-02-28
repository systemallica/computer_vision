import cv2

# Read video file
video = cv2.VideoCapture('video1.mp4')

# Check if camera opened successfully
if not video.isOpened():
    print("Error opening video stream or file")

# Get video duration
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
duration_seconds = frame_count/fps

# Read until video is completed
while video.isOpened():
    # Capture frame-by-frame
    ret, frame = video.read()

    if ret:
        # Get timestamp of current frame (in seconds)
        frame_timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
        # Resize the frame
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        # Switch between color and B&W
        if 1 < frame_timestamp < 2 or 3 < frame_timestamp < 4:
            cv2.putText(frame, 'B&W', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame_timestamp < 4:
            cv2.putText(frame, 'Color', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("frame", frame)

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