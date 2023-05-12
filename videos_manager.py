import cv2 as cv


class Video:
    grayscale_setting = False
    fourcc_code = None

    def __init__(self, grayscale, fourcc):
        self.grayscale_setting = grayscale
        self.fourcc_code = fourcc

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
