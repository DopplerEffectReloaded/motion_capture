import cv2 as cv
import os

class Video:

    def __init__(self, grayscale, filename, fourcc = '*DIVX'):
        self.grayscale_setting = grayscale
        self.fourcc_code = fourcc
        self.filename = filename

    def vid_store(self, bgr=True):
        """
        BGR is true by default
        If false is passed as argument it will store video file as hsv
        User can press space button to stop the video_storing
        :return: void method, allows you to store videos in the video format you want: BGR/HSV
        """
        fourcc = cv.VideoWriter_fourcc(*self.fourcc_code)
        out = cv.VideoWriter("output_eye.avi", fourcc, 24, (640, 480))
        capture_live_stream = cv.VideoCapture(0)
        while capture_live_stream.isOpened():
            retval, frame = capture_live_stream.read()  # Capturing frame-by-frame.

            if not retval:
                confirm = input("There was a problem in obtaining frame. Has the Stream ended? (Y/N)")
                if confirm == 'Y':
                    break
                else:
                    print("Please check your device. The camera may have closed.")
            if bgr:
                frame = cv.flip(frame, 1)
                out.write(frame)
                cv.imshow('frame', frame)
                if cv.waitKey(1) == ord(' '):
                    break
            else:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                frame = cv.flip(frame, 1)
                out.write(frame)
                cv.imshow('frame', frame)
                if cv.waitKey(1) == ord(' '):
                    break
        capture_live_stream.release()
        cv.destroyAllWindows()

    def vid_convert(filepath):

        filenames = sorted(i for i in os.listdir(filepath))
        if "frames" not in os.listdir():
            os.makedirs("frames")
        count = 0
        for filename in filenames:
            cap = cv.VideoCapture(filename)
            while True:

                _, frame = cap.read()
                if frame is None:
                    break
                cv.imwrite(filepath + "\\frame%d.jpg" % count, frame)
                count += 1
