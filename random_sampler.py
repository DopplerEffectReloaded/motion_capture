import cv2 as cv
import os


def flipper(path):
    
    image_names = [i for i in os.listdir(path)]
    
    count_flip = 0
    for name in image_names:
        img = cv.imread(path + "\\" + name)
        if count_flip == 2:
            count_flip = 0
            flipped = cv.flip(img, 1)
            cv.imwrite(path + "\\" + name, flipped)
        else:
            count_flip +=1
            continue

if __name__ == '__main__':
    flipper("Path to folder here")