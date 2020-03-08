import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
from random import randrange

resolution_x = 1024
resolution_y = 576
BW_VIDEO = 0
COLOR_VIDEO = 1


def process_video():
    input_path = '/Users/systemallica/Downloads/KU Leuven/Computer Vision/Assignment 1/input.mp4'
    output_path = '/Users/systemallica/Downloads/KU Leuven/Computer Vision/Assignment 1/output'

    # Read video file
    video = cv2.VideoCapture(input_path)

    # Check if camera opened successfully
    if not video.isOpened():
        print("Error opening video")

    basic_image_processing(video, output_path)
    object_detection(video, output_path)
    carte_blanche(video, output_path)
    join_videos(output_path)

    video.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def basic_image_processing(video, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out1 = cv2.VideoWriter(output_path + '/1.mp4', fourcc, 30.0, (resolution_x, resolution_y), COLOR_VIDEO)
    out2 = cv2.VideoWriter(output_path + '/2.mp4', fourcc, 30.0, (resolution_x, resolution_y), BW_VIDEO)
    out3 = cv2.VideoWriter(output_path + '/3.mp4', fourcc, 30.0, (resolution_x, resolution_y), COLOR_VIDEO)
    out4 = cv2.VideoWriter(output_path + '/4.mp4', fourcc, 30.0, (resolution_x, resolution_y), BW_VIDEO)
    out_blur = cv2.VideoWriter(output_path + '/5.mp4', fourcc, 30.0, (resolution_x, resolution_y), COLOR_VIDEO)
    out_grab = cv2.VideoWriter(output_path + '/6.mp4', fourcc, 30.0, (resolution_x, resolution_y), BW_VIDEO)

    # Read video
    while True:
        # Capture frame-by-frame
        ret, frame = video.read()

        # resize image
        frame = cv2.resize(frame, (resolution_x, resolution_y), interpolation=cv2.INTER_AREA)

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

            elif 3.75 > frame_timestamp > 3:
                # Black and white
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                out4.write(frame)

            elif 6 > frame_timestamp > 3.75:
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

            elif 11.5 > frame_timestamp > 10:
                # Bilateral smoothing
                frame = cv2.bilateralFilter(frame, 20, 100, 100)
                out_blur.write(frame)

            elif 16 > frame_timestamp > 11.5:
                # Grab red object in RGB space
                lower = [17, 15, 100]
                upper = [50, 56, 200]
                # create NumPy arrays from the boundaries
                lower = np.array(lower, dtype="uint8")
                upper = np.array(upper, dtype="uint8")
                # find the colors within the specified boundaries and apply the mask
                mask = cv2.inRange(frame, lower, upper)
                output = cv2.bitwise_and(frame, frame, mask=mask)
                output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                output = cv2.threshold(output, 10, 255, cv2.THRESH_BINARY)[1]
                out_grab.write(output)

            elif 18 > frame_timestamp > 16:
                # Grab red object(object in white, background in black)
                # Transform to HSV color domain
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Generate lower mask (0-5) and upper mask (175-180) of RED color
                mask_lower = cv2.inRange(frame_hsv, (0, 50, 20), (5, 255, 255))
                mask_upper = cv2.inRange(frame_hsv, (175, 50, 20), (180, 255, 255))

                # Merge the mask
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
                frame = cv2.bitwise_or(mask_lower, mask_upper)
                kernel = np.ones((5, 5), np.uint8)
                frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
                out_grab.write(frame)

            elif frame_timestamp > 20:
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
    out_sobel = cv2.VideoWriter(output_path + '/7.mp4', fourcc, 30.0, (resolution_x, resolution_y), BW_VIDEO)
    out_hough = cv2.VideoWriter(output_path + '/8.mp4', fourcc, 30.0, (resolution_x, resolution_y), COLOR_VIDEO)
    out_intensity = cv2.VideoWriter(output_path + '/9.mp4', fourcc, 30.0, (resolution_x, resolution_y), BW_VIDEO)

    # Read video
    while True:
        # Capture frame-by-frame
        ret, frame = video.read()

        # resize image
        frame = cv2.resize(frame, (resolution_x, resolution_y), interpolation=cv2.INTER_AREA)

        if ret:
            # Get timestamp of current frame (in seconds)
            frame_timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
            print(frame_timestamp)

            # Apply effect based on current timestamp
            if 21.25 > frame_timestamp > 20:
                edge_detection(out_sobel, frame, 1, 'horizontal')

            elif 22.75 > frame_timestamp > 21.25:
                edge_detection(out_sobel, frame, 3, 'horizontal')

            elif 23.5 > frame_timestamp > 22.75:
                edge_detection(out_sobel, frame, 3, 'vertical')

            elif 28 > frame_timestamp > 23.5:
                edge_detection(out_sobel, frame, 3, 'combined')

            elif 30 > frame_timestamp > 28:
                out_sobel.release()
                circle_detection(out_hough, frame, 1.2, 300, 50, 30, 100)

            elif 32.5 > frame_timestamp > 30:
                circle_detection(out_hough, frame, 1.2, 400, 100, 100, 200)

            elif 35.25 > frame_timestamp > 32.5:
                circle_detection(out_hough, frame, 1.2, 250, 120, 100, 180)

            elif 38.4 > frame_timestamp > 35.15:
                object_highlight(out_hough, frame, 1.2, 500)

            elif 41 > frame_timestamp > 38.4:
                out_hough.release()
                intensity_detection(out_intensity, frame, 1.2, 500)

            elif frame_timestamp > 41:
                break
        else:
            break

    out_intensity.release()


def intensity_detection(out, frame, dp, min_distance):
    # Remove noise
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    # Create copy of frame
    roi = frame.copy()
    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_distance, param1=40, param2=120, minRadius=130,
                               maxRadius=200)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # take the detected circle area as the roi
            roi = roi[y:y + r, x:x + r].copy()

        # roi is the object or region of object we need to find
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsvt = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # calculating object histogram
        histogram = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # normalize histogram and apply backprojection
        cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
        backprojection = cv2.calcBackProject([hsvt], [0, 1], histogram, [0, 180, 0, 256], 1)

        # show the output image
        out.write(backprojection)


def object_highlight(out, frame, dp, min_distance):
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_distance, param1=40, param2=120, minRadius=10,
                               maxRadius=80)
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
    # Sobel edge detection
    # Remove noise
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    # Converting to gray scale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect edges
    frame1 = cv2.Sobel(frame, cv2.CV_16S, 0, 1, ksize=k, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    frame2 = cv2.Sobel(frame, cv2.CV_16S, 1, 0, ksize=k, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    # Convert back to 8U so we can write it
    # Vertical
    abs_grad_y = cv2.convertScaleAbs(frame1)
    # Horizontal
    abs_grad_x = cv2.convertScaleAbs(frame2)

    if direction == 'vertical':
        out.write(abs_grad_y)
    elif direction == 'horizontal':
        out.write(abs_grad_x)
    else:
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        out.write(grad)


def circle_detection(out, frame, dp, min_distance, param2, min_radius, max_radius):
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_distance, param1=40, param2=param2, minRadius=min_radius,
                               maxRadius=max_radius)
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
    out = cv2.VideoWriter(output_path + '/10.mp4', fourcc, 30.0, (resolution_x, resolution_y), COLOR_VIDEO)
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

        # resize image
        frame = cv2.resize(frame, (resolution_x, resolution_y), interpolation=cv2.INTER_AREA)

        if ret:
            # Get timestamp of current frame (in seconds)
            frame_timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
            print(frame_timestamp)

            # Apply effect based on current timestamp
            if 10 > frame_timestamp >= 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = frame[y:y + h, x:x + w]
                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 20)
                    # I don't have more than two eyes
                    eyes = eyes[0:2]
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                out.write(frame)

            elif 14 > frame_timestamp > 10:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = frame[y:y + h, x:x + w]
                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 20)
                    # I don't have more than two eyes
                    eyes = eyes[0:2]
                    for (ex, ey, ew, eh) in eyes:
                        # Extract the eye region
                        sub_img = roi_color[ey:ey + eh, ex:ex + ew]

                        # Initialize black image of same dimensions for drawing the rectangles
                        blk = np.zeros(sub_img.shape, np.uint8)

                        # Draw colored rectangles with random color
                        cv2.rectangle(blk, (0, 0), (ex + ew, ey + eh), (randrange(256), randrange(256), randrange(256)), cv2.FILLED)

                        # Generate result by blending both images (opacity of rectangle image is 0.25 = 25 %)
                        res = cv2.addWeighted(sub_img, 1.0, blk, 0.5, 1)

                        roi_color[ey:ey+eh, ex:ex+ew] = res

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
