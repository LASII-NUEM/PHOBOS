import numpy as np
from datetime import datetime, timedelta
import os
import cv2
import math

videofile = 'recording_1764674952183.webm'

# look into the directory and save a frame each 5min
frame_direc = os.path.join("../PDI/frame2png_output/", videofile.split('.')[0])
files = os.listdir(frame_direc)
files_sorted = sorted(files, key=lambda x: int(x.split('_')[3].strip('.png')))

outFolder   = './ice_thickness_trace/'
os.makedirs(outFolder, exist_ok=True)

R_out = int(256)                # from hough detection radius
outer_diam_mm = 51/2
thickness_array = np.array([])
frametime = np.array([])

def draw_circle(event, x, y,flags, param):
    global drawing, thickness_array, cx, cy, R_out, outer_diam_mm

    base = param["base"]      # original image for redraw
    img  = param["img"]       # image shown in window (will be modified)
    win  = param["win"]
    file = param["file"]
    outFolder = param["outFolder"]

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        img[:] = base

        radius = int(math.hypot(x - cx, y - cy))
        cv2.circle(img, (cx, cy), radius, (0, 0, 255), 2)

        ice_thick = ((R_out-radius)*outer_diam_mm)/R_out
        thickness_array = np.append(thickness_array, ice_thick)

        print(f"Radius: {radius} [px]")
        print(f"ICE thickness: {ice_thick} [mm]")

        cv2.imshow(win, img)
        name = os.path.splitext(file)[0]
        frame_filename = os.path.join(outFolder, f"{name}_th{ice_thick:.2f}mm.png")
        cv2.imwrite(frame_filename, img)


for file in files_sorted:
    # ---------- UPLOAD FRAME ----------
    file_path = os.path.join(frame_direc, file)
    filename = file.split('_')[3].strip('.png')
    t = datetime.strptime(filename, '%H%M%S')
    frametime = np.append(frametime,t)

    f = cv2.imread(file_path)

    # ---------- DRAW OUTER CIRCLE ----------
    # define outer circle
    [w, h] = f.shape[:2]
    cx = w // 2
    cy = h // 2
    center_out = (int(cx), int(cy))
    cv2.drawMarker(f, center_out, (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
    cv2.circle(f, center_out, R_out, (255, 0, 0), 2)
    # cv2.imshow(file, f)
    # cv2.waitKey(0)

    # ---------- DRAW INNER CIRCLE ----------
    win = "Circle Window: " + str(file)

    img_show = f.copy()  # what you display
    base = f.copy()  # clean copy to redraw on every mouse move

    cv2.namedWindow(win)
    cv2.imshow(win, img_show)

    param = {
        "img": img_show,
        "base": base,
        "win": win,
        "file": file,
        "outFolder": outFolder
    }
    cv2.setMouseCallback(win, draw_circle, param)
    cv2.waitKey(0)

save_data = np.column_stack((frametime, thickness_array))
datapath = os.path.join(outFolder, "ice_thickness_vs_time.npy")
np.save(datapath, save_data)

