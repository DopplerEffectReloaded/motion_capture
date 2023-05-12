import os
import cv2 as cv
import numpy as np


class Images:
    images = []
    times = []
    lines = []

    def __init__(self, path):
        """
        Constructs an Images sequence objects with images of different exposure sequence
        This function assumes you have a text.txt file inside your folder.
        The text.txt file format can be found out from running the get format command.
        """
        images1 = []
        times1 = []
        with open(os.path.join(path, "text.txt")) as f:
            lines = f.readlines()
            self.lines = lines
            for line in lines:
                tokens = line.split()
                images1.append(cv.imread(os.path.join(path, tokens[0])))
                times1.append(1 / float(tokens[1]))
        self.images = images1
        self.times = np.array(times1, dtype=np.float32)

    def img_to_hdr(self):
        """
        Converts the series of images with different exposure into one fusion image
        """
        calibrate = cv.createCalibrateDebevec()
        response = calibrate.process(self.images, self.times)

        merge_debevec = cv.createMergeDebevec()
        hdr = merge_debevec.process(self.images, self.times, response)

        tonemap = cv.createTonemap(2.2)
        ldr = tonemap.process(hdr)

        merge_mertens = cv.createMergeMertens()
        fusion = merge_mertens.process(self.images)

        cv.imwrite("fusion1.png", fusion * 255)
        cv.imwrite("ldr.png", ldr * 255)
        cv.imwrite("hdr.hdr", hdr)

    def get_format(self):
        return self.lines


images = Images("C:\\Users\\SACHIN\\Video_Live_Stream\\motion_capture\\diff_exposure_seq")
print(images.get_format())
