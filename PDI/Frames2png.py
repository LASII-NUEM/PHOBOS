import numpy as np
from datetime import datetime, timedelta
import os
import cv2

def get_frames_as_numpy(filename):

    video = cv2.VideoCapture(filename)
    if not video.isOpened():
        print(f"Error: Could not open video {filename}")
        return

    while True:
        ret, frame = video.read()
        if not ret:
            break
        yield frame # frame is a numpy array (height, width, channels)
    video.release()


videofile = 'recording_1764674952183.webm'

videoFile_path = os.path.join('./data/Frames_input/', videofile)   # nome do arquivo de vídeo
outFolder   = os.path.join('./PDI/frame2png_output/', videofile.split('.')[0])          # pasta de saída

os.makedirs(outFolder, exist_ok=True)

startdateStr = '01/12/2025'
startTimeStr = '11:26:44'
t1 = "01/12/2025 11:26:44"
f1 = "%d/%m/%Y %H:%M:%S"
# formato D/M/Y HH:MM:SS
t0 = datetime.strptime(t1,f1 )

firstFrameIdx = 11
fps = 30
skipmin = 5                     # 1 frame / 5 min
stepFrames    = fps*skipmin*60
frameGlobalIdx = 0
savedIdx       = 0
wantedframes = 20

kernel_size = 11                    # Kernel size must be positive and odd
sigma = 2.5                        # 0 means sigma is calculated from kernel size
gaussian_kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)
gaussian_kernel_2d = np.outer(gaussian_kernel_1d, gaussian_kernel_1d.transpose())

for frame in get_frames_as_numpy(videoFile_path):
    frameGlobalIdx +=1
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #frame to gray scale
    # cv2.imshow('Grayscale Frame: ' + str(frameGlobalIdx), gray_frame)
    # cv2.waitKey(0)

    if frameGlobalIdx < firstFrameIdx:
        continue

    k = frameGlobalIdx - firstFrameIdx
    if k==0:
        filtered_gray_frame = cv2.filter2D(gray_frame, -1, gaussian_kernel_2d)
        # cv2.imshow('Filtered Frame: ' + str(frameGlobalIdx), filtered_gray_frame)
        # cv2.waitKey(0)

        circles = cv2.HoughCircles(
            filtered_gray_frame,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,  # Minimum distance between centers
            param1=50,
            param2=30,  # Sensitivity (lower = more circles)
            minRadius=250,
            maxRadius=300
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            x, y, r = circles[0, 0]
            center = (int(x), int(y))
            radius = int(r)

            circle_frame = frame.copy()
            circle_roi = cv2.circle(circle_frame, center, radius, color=(0, 0, 255), thickness=2)
            # cv2.imshow('Hough circle id' + str(frameGlobalIdx), circle_roi)
            # cv2.waitKey(0)

        side_length = int((2 * radius)+20)
        half_side = int(side_length / 2)
        top_left = (center[0] - half_side, center[1] - half_side)
        bottom_right = (center[0] + half_side, center[1] + half_side)

        squared_roi = cv2.rectangle(circle_frame, top_left, bottom_right, color = (0, 255, 0), thickness=2)
        # cv2.imshow("Circle with Square", squared_roi)
        # cv2.waitKey(0)

        h, w = frame.shape[:2]
        crop_frame = frame[
            max(center[1] - half_side, 0): min(center[1] + half_side, h),
            max(center[0] - half_side, 0): min(center[0] + half_side, w)
        ].copy()
        # cv2.imshow("Croped frame", crop_frame)
        # cv2.waitKey(0)

        # If roi is BGR, convert to gray first:
        if crop_frame.ndim == 3:
            crop_frame_gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
        else:
            crop_frame_gray = crop_frame

        # Otsu thresholding
        _, roi_bw = cv2.threshold(
            crop_frame_gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # cv2.imshow("Otsu threshold", roi_bw)
        # cv2.waitKey(0)

        radius = 7
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        roi_morf = cv2.morphologyEx(roi_bw, cv2.MORPH_CLOSE, se, iterations=1)
        # cv2.imshow("Morphology", roi_morf)
        # cv2.waitKey(0)

        # Hadamard product
        mask01 = (roi_morf > 0).astype(np.float64)
        roi_masked = crop_frame_gray.astype(np.float64) * mask01
        roi_masked_u8 = np.clip(roi_masked, 0, 255).astype(np.uint8)
        # cv2.imshow("Hadamard product", roi_masked_u8)
        # cv2.waitKey(0)

    else:
        h,w = frame.shape[:2]
        crop_frame = gray_frame[
            max(center[1] - half_side, 0): min(center[1] + half_side, h),
            max(center[0] - half_side, 0): min(center[0] + half_side, w)
        ].copy()
        roi_masked = crop_frame.astype(np.float64) * mask01
        roi_masked_u8 = np.clip(roi_masked, 0, 255).astype(np.uint8)
        # cv2.imshow("Hadamard set for frame:" + str(frameGlobalIdx), roi_masked_u8)
        # cv2.waitKey(0)

    # Save frame as .png
    if k % stepFrames != 0:
        continue


    if wantedframes == savedIdx:
        continue

    frameTime = datetime.strftime(t0 + timedelta(seconds=(frameGlobalIdx-firstFrameIdx)/fps), "%d%m%Y_%H%M%S")
    frame_filename = os.path.join(outFolder, f"frame_{frameGlobalIdx:04d}_{frameTime}.png")
    cv2.imwrite(frame_filename, roi_masked_u8)
    cv2.destroyAllWindows()
    print(f"frame {frameGlobalIdx} saved!")

    savedIdx += 1
    pass

print(f"\n Done! A total of {savedIdx} frames saved!\n")
print(f"Frame directory: {outFolder}\n")