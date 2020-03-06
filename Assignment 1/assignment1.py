import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips


def process_video():
    output_path = '/Users/systemallica/Downloads/KU Leuven/Computer Vision/Assignment 1/output'
    input_path_1 = 'video3.mp4'
    input_path_2 = 'video4.mp4'

    # Read video file
    video_1 = cv2.VideoCapture(input_path_1)
    video_2 = cv2.VideoCapture(input_path_2)

    # Check if camera opened successfully
    if not video_1.isOpened() or not video_2.isOpened():
        print("Error opening video")

    basic_image_processing(video_1, output_path)
    object_detection(video_1, output_path)
    carte_blanche(video_2, output_path)
    join_videos(output_path)

    video_1.release()
    video_2.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def basic_image_processing(video, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out1 = cv2.VideoWriter(output_path + '/1.mp4', fourcc, 30.0, (1024, 576), 1)
    out2 = cv2.VideoWriter(output_path + '/2.mp4', fourcc, 30.0, (1024, 576), 0)
    out3 = cv2.VideoWriter(output_path + '/3.mp4', fourcc, 30.0, (1024, 576), 1)
    out4 = cv2.VideoWriter(output_path + '/4.mp4', fourcc, 30.0, (1024, 576), 0)
    out_blur = cv2.VideoWriter(output_path + '/5.mp4', fourcc, 30.0, (1024, 576), 1)
    out_grab = cv2.VideoWriter(output_path + '/6.mp4', fourcc, 30.0, (1024, 576), 0)

    # Read video
    while True:
        # Capture frame-by-frame
        ret, frame = video.read()

        width = int(1024)
        height = int(576)
        dim = (width, height)
        # resize image
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        if ret:
            # Get timestamp of current frame (in seconds)
            frame_timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
            print(frame_timestamp)

            # Apply effect based on current timestamp
            if 1 > frame_timestamp >= 0:
                # Color
                out1.write(frame)

            elif 2 > frame_timestamp > 1:
                # Black and white
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                out2.write(frame)

            elif 3 > frame_timestamp > 2:
                # Color
                out3.write(frame)

            elif 4 > frame_timestamp > 3:
                # Black and white
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                out4.write(frame)

            elif 6 > frame_timestamp > 4:
                # Gaussian smoothing
                kernel_size = 11
                frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
                out_blur.write(frame)

            elif 8 > frame_timestamp > 6:
                # Gaussian smoothing
                kernel_size = 19
                frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
                out_blur.write(frame)

            elif 10 > frame_timestamp > 8:
                # Gaussian smoothing
                kernel_size = 29
                frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
                out_blur.write(frame)

            elif 12 > frame_timestamp > 10:
                # Bilateral smoothing
                frame = cv2.bilateralFilter(frame, 20, 100, 100)
                out_blur.write(frame)

            elif 16 > frame_timestamp > 12:
                # Grab red object in RGB(object in white, background in black
                # Transform to grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Apply binary threshold
                ret, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY_INV)
                out_grab.write(frame)

            elif 18 > frame_timestamp > 16:
                # Grab red object(object in white, background in black)
                # Transform to HSV color domain
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Generate lower mask (0-5) and upper mask (175-180) of RED color
                mask_lower = cv2.inRange(frame_hsv, (0, 50, 20), (5, 255, 255))
                mask_upper = cv2.inRange(frame_hsv, (175, 50, 20), (180, 255, 255))

                # Merge the mask
                # It is a B&W image
                frame = cv2.bitwise_or(mask_lower, mask_upper)
                out_grab.write(frame)

            elif 20 > frame_timestamp > 18:
                # Grab red object(object in white, background in black)
                # Transform to HSV color domain
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Generate lower mask (0-5) and upper mask (175-180) of RED color
                mask_lower = cv2.inRange(frame_hsv, (0, 50, 20), (5, 255, 255))
                mask_upper = cv2.inRange(frame_hsv, (175, 50, 20), (180, 255, 255))

                # Merge the mask
                # It is a B&W image
                frame = cv2.bitwise_or(mask_lower, mask_upper)
                kernel = np.ones((5, 5), np.uint8)
                frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
                out_grab.write(frame)

            else:
                break
        else:
            break

    out1.release()
    out2.release()
    out3.release()
    out4.release()
    out_blur.release()
    out_grab.release()


def object_detection(video, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_sobel = cv2.VideoWriter(output_path + '/7.mp4', fourcc, 30.0, (1024, 576), 0)
    out_hough = cv2.VideoWriter(output_path + '/8.mp4', fourcc, 30.0, (1024, 576), 1)
    out_intensity = cv2.VideoWriter(output_path + '/9.mp4', fourcc, 30.0, (1024, 576), 1)

    # Read video
    while True:
        # Capture frame-by-frame
        ret, frame = video.read()

        width = int(1024)
        height = int(576)
        dim = (width, height)
        # resize image
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        if ret:
            # Get timestamp of current frame (in seconds)
            frame_timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
            print(frame_timestamp)

            # Apply effect based on current timestamp
            if 21.25 > frame_timestamp > 20:
                edge_detection(out_sobel, frame, 3, 'horizontal')

            elif 22.75 > frame_timestamp > 21.25:
                edge_detection(out_sobel, frame, 5, 'horizontal')

            elif 23.5 > frame_timestamp > 22.75:
                edge_detection(out_sobel, frame, 3, 'vertical')

            elif 25 > frame_timestamp > 23.5:
                edge_detection(out_sobel, frame, 5, 'vertical')

            elif 27.5 > frame_timestamp > 25:
                circle_detection(out_hough, frame, 1.2, 300)

            elif 30 > frame_timestamp > 27.5:
                circle_detection(out_hough, frame, 1.2, 400)

            elif 32.5 > frame_timestamp > 30:
                circle_detection(out_hough, frame, 1.2, 450)

            elif 35 > frame_timestamp > 32.5:
                circle_detection(out_hough, frame, 1.2, 500)

            elif 37 > frame_timestamp > 35:
                object_highlight(out_hough, frame, 1.2, 400)

            elif 39.8 > frame_timestamp > 37:
                intensity_detection(out_intensity, frame, 1.2, 400)

            elif frame_timestamp > 39.8:
                break
        else:
            break

    out_sobel.release()
    out_hough.release()
    out_intensity.release()


def intensity_detection(out, frame, dp, min_distance):
    # resize to 'square' image so intensity detection works
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    roi = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_distance, param1=40, param2=25, minRadius=40,
                               maxRadius=200)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw a rectangle around the detected circle
            roi = roi[y:y + r, x:x + r].copy()

        # roi is the object or region of object we need to find
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # target is the image we search in
        target = frame
        hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

        # calculating object histogram
        M = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # normalize histogram and apply backprojection
        cv2.normalize(M, M, 0, 255, cv2.NORM_MINMAX)
        B = cv2.calcBackProject([hsvt], [0, 1], M, [0, 180, 0, 256], 1)
        cv2.normalize(B, B, 0, 255, cv2.NORM_MINMAX)
        # Use thresholding to segment out the region
        ret, thresh = cv2.threshold(B, 10, 255, 0)

        # Overlay images using bitwise_and
        thresh = cv2.merge((thresh, thresh, thresh))
        res = cv2.bitwise_and(target, thresh)

        # Resize back to original size
        res = cv2.resize(res, (1024, 576), interpolation=cv2.INTER_AREA)

        # show the output image
        out.write(res)


def object_highlight(out, frame, dp, min_distance):
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_distance, param1=40, param2=25, minRadius=40,
                               maxRadius=200)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw a rectangle around the detected circle
            cv2.rectangle(output, (x - (r + 5), y - (r + 5)), (x + (r + 5), y + (r + 5)), (0, 255, 0), 2)
        # show the output image
        out.write(output)


def edge_detection(out, frame, k, direction):
    # Sobel horizontal edge detection
    # Converting to gray scale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Remove noise
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    # Detect edges
    frame1 = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=k)
    frame2 = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=k)

    # Take absolute and convert back to 8U so we can write it
    # Vertical
    frame1 = np.absolute(frame1)
    frame1 = np.uint8(frame1)
    # Horizontal
    frame2 = np.absolute(frame2)
    frame2 = np.uint8(frame2)

    if direction == 'vertical':
        out.write(frame1)
    elif direction == 'horizontal':
        out.write(frame2)


def circle_detection(out, frame, dp, min_distance):
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_distance, param1=40, param2=25, minRadius=40,
                               maxRadius=200)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        # show the output image
        out.write(output)


def carte_blanche(video, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path + '/10.mp4', fourcc, 30.0, (1024, 576), 1)
    # Pre-trained Cascade Classifier models for face and eye detection
    # Path depends on OpenCV installation directory
    face_cascade_name = '/Users/systemallica/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml'
    eyes_cascade_name = '/Users/systemallica/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_name)
    eye_cascade = cv2.CascadeClassifier(eyes_cascade_name)

    # Read video
    while True:
        # Capture frame-by-frame
        ret, frame = video.read()

        width = int(1024)
        height = int(576)
        dim = (width, height)
        # resize image
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        if ret:
            # Get timestamp of current frame (in seconds)
            frame_timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
            print(frame_timestamp)

            # Apply effect based on current timestamp
            if 14 > frame_timestamp >= 0:
                # detect_face(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = frame[y:y + h, x:x + w]
                    # Change eye color by changing color space
                    roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 20)
                    # I don't have more than two eyes
                    eyes = eyes[0:2]
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                out.write(frame)

            elif frame_timestamp > 14:
                break

    out.release()
    cv2.destroyAllWindows()


def join_videos(path):
    files = []
    videos = []

    # Get a list of the files in the directory
    for r, d, f in os.walk(path):
        for file in f:
            if file == '.DS_Store':
                continue
            files.append(os.path.join(r, file))

    # Sort by name
    files = sorted(files, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

    # Create a VideoClip for each file
    for file in files:
        video = VideoFileClip(file)
        videos.append(video)

    # Concatenate all videos in order and write the output
    final_clip = concatenate_videoclips(videos)
    final_clip.write_videofile("final.mp4")


process_video()
