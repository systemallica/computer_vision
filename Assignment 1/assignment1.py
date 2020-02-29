import cv2
import numpy as np


def process_video():
    # Read video file
    video = cv2.VideoCapture('video2.mp4')

    # Check if camera opened successfully
    if not video.isOpened():
        print("Error opening video")

    part1(video)
    part2(video)

    # Closes all the frames
    cv2.destroyAllWindows()


def part1(video):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out1 = cv2.VideoWriter('output1.mp4', fourcc, 30.0, (1920, 1080), 1)
    out2 = cv2.VideoWriter('output2.mp4', fourcc, 30.0, (1920, 1080), 0)
    out3 = cv2.VideoWriter('output3.mp4', fourcc, 30.0, (1920, 1080), 1)
    out4 = cv2.VideoWriter('output4.mp4', fourcc, 30.0, (1920, 1080), 0)
    out_blur = cv2.VideoWriter('output5.mp4', fourcc, 30.0, (1920, 1080), 1)
    out_grab = cv2.VideoWriter('output6.mp4', fourcc, 30.0, (1920, 1080), 0)

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


def part2(video):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output7.mp4', fourcc, 30.0, (1920, 1080), 1)

    # Read video
    while True:
        # Capture frame-by-frame
        ret, frame = video.read()

        if ret:
            # Get timestamp of current frame (in seconds)
            frame_timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000

            # Apply effect based on current timestamp
            if 25 > frame_timestamp > 20:
                # Color
                add_subtitle(frame, 'Color', 3)
                out.write(frame)
            else:
                break
        else:
            break

    video.release()
    out.release()


def add_subtitle(frame, text, size):
    cv2.putText(frame, text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 255, 0), 2, cv2.LINE_AA)


process_video()
