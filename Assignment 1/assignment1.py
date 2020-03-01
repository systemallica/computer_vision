import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips


def process_video():

    output_path = '/Users/systemallica/Downloads/KU Leuven/Computer Vision/Assignment 1/output'
    input_path = 'video3.mp4'

    # Read video file
    video = cv2.VideoCapture(input_path)

    # Check if camera opened successfully
    if not video.isOpened():
        print("Error opening video")

    basic_image_processing(video, output_path)
    object_detection(video, output_path)
    join_videos(output_path)

    video.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def basic_image_processing(video, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out1 = cv2.VideoWriter(output_path + '/1.mp4', fourcc, 30.0, (1920, 1080), 1)
    out2 = cv2.VideoWriter(output_path + '/2.mp4', fourcc, 30.0, (1920, 1080), 0)
    out3 = cv2.VideoWriter(output_path + '/3.mp4', fourcc, 30.0, (1920, 1080), 1)
    out4 = cv2.VideoWriter(output_path + '/4.mp4', fourcc, 30.0, (1920, 1080), 0)
    out_blur = cv2.VideoWriter(output_path + '/5.mp4', fourcc, 30.0, (1920, 1080), 1)
    out_grab = cv2.VideoWriter(output_path + '/6.mp4', fourcc, 30.0, (1920, 1080), 0)

    # Read video
    while True:
        # Capture frame-by-frame
        ret, frame = video.read()

        if ret:
            # Get timestamp of current frame (in seconds)
            frame_timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000

            # Apply effect based on current timestamp
            if 1 > frame_timestamp >= 0:
                # Color
                add_subtitle(frame, 'Color', 3)
                out1.write(frame)

            elif 2 > frame_timestamp > 1:
                # Black and white
                add_subtitle(frame, 'B&W', 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                out2.write(frame)

            elif 3 > frame_timestamp > 2:
                # Color
                add_subtitle(frame, 'Color', 3)
                out3.write(frame)

            elif 4 > frame_timestamp > 3:
                # Black and white
                add_subtitle(frame, 'B&W', 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                out4.write(frame)

            elif 6 > frame_timestamp > 4:
                # Gaussian smoothing
                kernel_size = 11
                frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
                add_subtitle(frame, 'Gaussian Blur. Kernel size: ' + str(kernel_size), 1)
                out_blur.write(frame)

            elif 8 > frame_timestamp > 6:
                # Gaussian smoothing
                kernel_size = 19
                frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
                add_subtitle(frame, 'Gaussian Blur. Kernel size: ' + str(kernel_size), 1)
                out_blur.write(frame)

            elif 10 > frame_timestamp > 8:
                # Gaussian smoothing
                kernel_size = 29
                frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
                add_subtitle(frame, 'Gaussian Blur. Kernel size: ' + str(kernel_size), 1)
                out_blur.write(frame)

            elif 12 > frame_timestamp > 10:
                # Bilateral smoothing
                frame = cv2.bilateralFilter(frame, 20, 100, 100)
                add_subtitle(frame, 'Bilateral filter(edges are preserved)', 1)
                out_blur.write(frame)

            elif 16 > frame_timestamp > 12:
                # Grab red object in RGB(object in white, background in black
                # Transform to grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Apply binary threshold
                ret, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY_INV)
                add_subtitle(frame, 'Grabbing red object RGB', 1)
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
                add_subtitle(frame, 'Grabbing red object HSV', 1)
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
                add_subtitle(frame, 'Improved grabbing red object HSV', 1)
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
    out_sobel = cv2.VideoWriter(output_path + '/7.mp4', fourcc, 30.0, (1920, 1080), 0)
    out_hough = cv2.VideoWriter(output_path + '/8.mp4', fourcc, 30.0, (1920, 1080), 1)

    # Read video
    while True:
        # Capture frame-by-frame
        ret, frame = video.read()

        if ret:
            # Get timestamp of current frame (in seconds)
            frame_timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
            print(frame_timestamp)

            # Apply effect based on current timestamp
            if 22.5 > frame_timestamp > 20:
                edge_detection(out_sobel, frame, 5, 'combined')

            elif 25 > frame_timestamp > 22.5:
                edge_detection(out_sobel, frame, 11, 'combined')

            elif 27.5 > frame_timestamp > 25:
                circle_detection(out_hough, frame, 1.2, 100)

            elif 30 > frame_timestamp > 27.5:
                circle_detection(out_hough, frame, 1.2, 200)

            elif 32.5 > frame_timestamp > 30:
                circle_detection(out_hough, frame, 1.2, 300)

            elif 35 > frame_timestamp > 32.5:
                circle_detection(out_hough, frame, 1.2, 400)

            elif frame_timestamp > 35:
                break
        else:
            break

    out_sobel.release()
    out_hough.release()


def edge_detection(out, frame, k, direction):
    # Sobel horizontal edge detection
    # converting to gray scale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # remove noise
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    frame1 = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=k)
    frame2 = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=k)

    # Take absolute and convert back to 8U so we can write it
    # Vertical
    frame1 = np.absolute(frame1)
    frame1 = np.uint8(frame1)
    # Horizontal
    frame2 = np.absolute(frame2)
    frame2 = np.uint8(frame2)
    # Combined
    frame = frame1 + frame2

    if direction == 'vertical':
        add_subtitle(frame1, 'Sobel vertical edge detection', 1)
        out.write(frame1)
    elif direction == 'horizontal':
        add_subtitle(frame2, 'Sobel horizontal edge detection', 1)
        out.write(frame2)
    else:
        add_subtitle(frame2, 'Sobel both directions edge detection', 1)
        out.write(frame)


def circle_detection(out, frame, dp, min_distance):
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_distance, param1=40, param2=25, minRadius=40, maxRadius=200)
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


def join_videos(path):
    files = []
    videos = []

    # Get a list of the files in the directory
    for r, d, f in os.walk(path):
        for file in f:
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


def add_subtitle(frame, text, size):
    cv2.putText(frame, text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 255, 0), 2, cv2.LINE_AA)


process_video()
