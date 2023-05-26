from __future__ import print_function
from __future__ import division
import numpy as np
import cv2 as cv
import sys
import os

CENTER_COORDS_LIST = [(630, 10), (10, 10), (10, 471), (630, 471)]
RADIUS_OBJ = 10
START = ()
# CHANGE_PHASE = False
END = ()
def image_render(imgname):
    img = cv.imread(imgname)
    if img is None:
        sys.exit("Could not read image")
    cv.namedWindow("Starry Night", cv.WINDOW_NORMAL)
    cv.imshow("Starry Night", img)
    if cv.waitKey(0) == ord("q"):
        cv.imwrite("castle.png", img)


def select_region():
    image = cv.imread("starry_night.png")
    star_section = image[450:1550, 650:2000]
    image[:, :, 2] = 0
    cv.namedWindow("Starry Night", cv.WINDOW_NORMAL)
    cv.imshow("Starry Night", image)
    cv.waitKey(0)
    cv.imwrite("star_section.png", image)


def image_blending():
    img1 = cv.imread("starry_night.png")
    img2 = cv.imread("castle.png")
    assert img1 is not None, "file could not be opened"
    assert img2 is not None, "file could not be opened"
    img_comb = cv.addWeighted(img1, 0.4, img2, 0.7, 0)
    cv.namedWindow("Taj Mahal under Stars", cv.WINDOW_NORMAL)
    cv.imshow("Taj Mahal under Stars", img_comb)
    cv.waitKey(0)
    cv.destroyAllWindows()


def image_adder():
    # Load two images
    img1 = cv.imread('castle.png')
    img2 = cv.imread('starry_night.png')
    assert img1 is not None, "file could not be read, check with os.path.exists()"
    assert img2 is not None, "file could not be read, check with os.path.exists()"
    # I want to put logo on top-left corner, So I create a ROI
    rows, cols, channels = img2.shape
    rows = int(rows / 2)
    cols = int(cols / 2)

    roi = img1[0:rows, 0:cols]
    img2 = img2[0:rows, 0:cols]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img2gray, 80, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv.bitwise_and(img2, img2, mask=mask)
    # Put logo in ROI and modify the main image
    dst = cv.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst
    cv.namedWindow('res', cv.WINDOW_NORMAL)
    cv.imshow('res', img1)
    cv.waitKey(0)
    cv.destroyAllWindows()


def object_extract():
    cap = cv.VideoCapture("output.avi")
    while True:
        retval, frame = cap.read()

        if not retval:
            confirm = input("There was a problem in obtaining stream. Has Stream ended? (Y/N)")
            if confirm == "Y":
                break
            else:
                print("Please check your camera.")
        hsv_yellow = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hsv_blue = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hsv_pink = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # defining the colour range for yellow
        upper_yellow = np.array([30, 255, 255])
        lower_yellow = np.array([20, 50, 50])

        # defining the colour range for blue
        upper_blue = np.array([110, 255, 255])
        lower_blue = np.array([100, 50, 50])

        # defining the colour range for pink
        upper_pink = np.array([170, 255, 255])
        lower_pink = np.array([160, 50, 50])

        # creating masks
        mask_pink = cv.inRange(hsv_pink, lower_pink, upper_pink)
        mask_blue = cv.inRange(hsv_blue, lower_blue, upper_blue)
        mask_yellow = cv.inRange(hsv_yellow, lower_yellow, upper_yellow)

        res_yellow = cv.bitwise_and(frame, frame, mask=mask_yellow)
        res_blue = cv.bitwise_and(frame, frame, mask=mask_blue)
        res_pink = cv.bitwise_and(frame, frame, mask=mask_pink)

        dummy = cv.add(res_pink, res_blue)
        final = cv.add(dummy, res_yellow)

        cv.imshow('frame', frame)
        cv.imshow('Obj Tracked', final)

        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break
    cv.destroyAllWindows()


def load_exposure_seq(path):
    images1 = []
    times1 = []
    with open(os.path.join(path, "text.txt")) as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split()
            images1.append(cv.imread(os.path.join(path, tokens[0])))
            times1.append(1 / float(tokens[1]))
    return images1, np.array(times1, dtype=np.float32)


def img_to_hdr(images_param, times_param):
    calibrate = cv.createCalibrateDebevec()
    response = calibrate.process(images_param, times_param)

    merge_debevec = cv.createMergeDebevec()
    hdr = merge_debevec.process(images_param, times_param, response)

    tonemap = cv.createTonemap(2.2)
    ldr = tonemap.process(hdr)

    merge_mertens = cv.createMergeMertens()
    fusion = merge_mertens.process(images_param)

    cv.imwrite("fusion.png", fusion * 255)
    cv.imwrite("ldr.png", ldr * 255)
    cv.imwrite("hdr.hdr", hdr)


def object_tracker(alg_name):
    """
    This method uses background subtraction to track objects
    :param alg_name: to use
    :return: null
    It is a void method
    """
    cap = cv.VideoCapture("sample_video.avi")
    if alg_name == "MOG2":
        backSub = cv.createBackgroundSubtractorMOG2()
    else:
        backSub = cv.createBackgroundSubtractorKNN()
    while True:
        retval, frame = cap.read()

        if not retval:
            confirm = input("There was a problem in obtaining stream. Has Stream ended? (Y/N)")
            if confirm == "Y":
                break
            else:
                print("Please check your camera.")
                break

        fgMask = backSub.apply(frame)

        cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow('frame', frame)
        cv.imshow('fgmask', fgMask)

        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break
    cv.destroyAllWindows()


def object_tracker_meanshift():
    """
    Overloaded function differs in parameters it accepts
    Uses meanshift tracking
    """
    cap = cv.VideoCapture("output_eye.avi")

    retval, frame = cap.read()

    x, y, w, h = 300, 200, 100, 50
    track_window = (x, y, w, h)

    roi = frame[y:y + h, x:x + w]
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 100, 1)
    while True:
        retval, frame = cap.read()

        if retval:
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            ret, track_window = cv.meanShift(dst, track_window, term_crit)

            x, y, w, h = track_window
            img2 = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 0)
            cv.imshow('Img', img2)

            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break


def obj_tracker():
    backSub = cv.createBackgroundSubtractorMOG2()
    global good_new
    global good_old
    cap = cv.VideoCapture("output_eye.avi")
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    fgMask = backSub.apply(old_frame)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=fgMask, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    while 1:
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv.add(frame, mask)
        cv.imshow('frame', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    cv.destroyAllWindows()


def movingobj_tracker():
    cap = cv.VideoCapture("sample_video.avi")
    ret, frame1 = cap.read()
    frame_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    while 1:
        ret, frame2 = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        next_frame = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(frame_gray, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2  # converting angle to degrees from rad in math formula is ang(deg)/(2*pi)
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        print(hsv[..., 2])
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imshow('frame2', bgr)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv.imwrite('opticalfb.png', frame2)
            cv.imwrite('opticalhsv.png', bgr)
        frame_gray = next_frame
    cv.destroyAllWindows()


def detect_and_display(frame, face_cascade, eyes_cascade, frame_count, change_phase):
    """
    This is a helper method for detecting face and eyes per frame
    To be used in the main function eye_face_tracker
    :param frame: to detect
    :param face_cascade: Face haar cascade
    :param eyes_cascade: Eyes haar cascade
    :return: void method displays frame
    """
    frame = cv.flip(frame, 1)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    # Detecting faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)

        faceROI = frame_gray[y:y + h, x:x + w]
        # faceROI1 = frame[y:y+h, x: x + w]
        # cv.imshow('Capture-Face Detection', faceROI1)
        # if cv.waitKey(0) == ord(' '):
        #     break
        # Detecting eyes in each face

        eyes = eyes_cascade.detectMultiScale(faceROI)
        # eye_right=False 
        eye_left = ()
        eye_right = ()
        flag = True
        for (x2, y2, w2, h2) in eyes:
            eyes_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            if flag:
                eye_left = eyes_center
                flag = False
            else:
                eye_right = eyes_center
                flag=True
            if eye_left==():
                continue
            elif eye_right==():
                continue
            else:
                both_eyes = (eye_left, eye_right)
            point = tuple([int(sum(q)/len(q)) for q in zip(*both_eyes)])

            if frame_count == 0:
                START = point
                print(START)
                
            if change_phase:
                END = point
                print(END)
            
            # temp = (eye_left_lst[i], eye_right_lst[i])
            # center_coords = [sum(q)/len(q) for q in zip(*temp)]
            # print(type(center_coords))
            # print(eyes_center)
            radius = int(round((w2 + h2) * 0.25))

            # eye_ROI = faceROI1[y2:y2+h2, x2:x2+w2]
            # row, col, _ = eye_ROI.shape
            # waste = cv.dilate(eye_ROI, np.ones((5,5), dtype=np.uint8))
            # waste = cv.GaussianBlur(waste, (5,5), 1)
            # eye_ROI_without_waste = 255 - cv.absdiff(eye_ROI, waste)
            # print(eye_ROI)
            # eye_gauss = cv.cvtColor(eye_ROI, cv.COLOR_BGR2GRAY)
            # eye_gauss = cv.GaussianBlur(eye_gauss, (7, 7), 0)
            # _, thresh = cv.threshold(eye_gauss, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
            # thresh = cv.adaptiveThreshold(eye_gauss, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV, 11,2)
            # cv.imshow("Threshold", thresh)
            # key = cv.waitKey(30)
            # if key == 27:
            #     break
            # contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            
            # contours = sorted(contours, key=lambda x:cv.contourArea(x), reverse=True)
            
            # for contour in contours:
            #     (a, b, c, d) = cv.boundingRect(contour)
            #     frame = cv.rectangle(frame, (x2+w2+a, y2+h2+b), (x2+w2+a+c, y2+h2+b+d), (255, 0, 0), 2)
            #     frame = cv.line(frame, (a+c//2, 0), (a+c//2, row), (0,255,0),2)
            #     frame = cv.line(frame, (0, b+d//2), (col, b+d//2), (0, 0, 255), 2)
            #     break
            # cv.imshow('Capture eye', eye_ROI)
            # if cv.waitKey(0) == ord(' '):
            #     break

            
            frame = cv.circle(frame, point, radius=3, color=(0,0,255), thickness=-1)
            frame = cv.circle(frame, eyes_center, radius, (255, 0, 0), 4)
    cv.namedWindow('Capture-Face Detection', cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty('Capture-Face Detection', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow('Capture-Face Detection', frame)


def eye_and_face_tracker():
    """
    This is a caller method which calls the above function per frame
    :return: void method
    """
    face_cascade_name = "C:\\Users\\SACHIN\\anaconda3\\Lib\\site-packages\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml"  # Path
    eyes_cascade_name = "C:\\Users\\SACHIN\\anaconda3\\Lib\\site-packages\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml"  # Path
    face_cascade = cv.CascadeClassifier()
    eyes_cascade = cv.CascadeClassifier()
    if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)
    if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
        print('--(!)Error loading eyes cascade')
        exit(0)
    # fourcc = cv.VideoWriter_fourcc(*'DIVX')
    # out = cv.VideoWriter("output_eye.avi", fourcc, 7, (640, 480))
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print('--(!)Error opening camera')
        exit(0)
    frame_count = -1
    change_phase = False
    while True:
        ret, frame = cap.read()
        if ret is None:
            break
        # frame = cv.line(frame, (335, 0), (335,512), (0,0,255), 4)
        # frame = cv.line(frame, (0, 250), (1000,250), (0,0,255), 4)
        
        frame_count += 1
        if frame_count == 50:
            change_phase = True
        if frame_count == 100:
            change_phase = True
        if frame_count == 150:
            change_phase = True
        if frame_count//50 <= 3:
            frame = cv.circle(frame, CENTER_COORDS_LIST[frame_count//50], RADIUS_OBJ, color=(0,255,255), thickness=-1)
        else:
            frame_count=1
            change_phase = True
        detect_and_display(frame, face_cascade, eyes_cascade, frame_count, change_phase)
        change_phase = False
        # out.write(frame)
        if cv.waitKey(50) == ord(' '):
            break

if __name__ == '__main__':
    eye_and_face_tracker()