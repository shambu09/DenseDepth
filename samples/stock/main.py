import cv2
import numpy as np
import glob

# Preprocess
# The images were of four channels, discarded the fourth channel.

g = glob.glob("*.png")

for File in g:
    img = cv2.imread(File)
    img = img[:, :, 0:3]
    img = img[0:128, 0:128, :]
    cv2.imwrite(File, img)
