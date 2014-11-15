import cv2
import numpy as np

class Polygon(object):
    def __init__(self, points, red=0, blue=0, green=0, opacity=1.0):
        self.points = points
        self.red = red
        self.blue = blue
        self.green = green
        self.opacity = opacity

    def draw_on_image(self, img):
        poly = np.array(self.points, np.int32)
        w, h, _ = img.shape
        mask = np.zeros((w, h))
        cv2.fillPoly(mask, [poly], 1)

        #opacity
        for i in xrange(w):
            for j in xrange(h):
                #polygon exists there
                if mask[i][j]:
                    b0, g0, r0 = img[i][j]
                    r1 = (1.0 - self.opacity) * r0 + self.opacity * self.red
                    g1 = (1.0 - self.opacity) * g0 + self.opacity * self.green
                    b1 = (1.0 - self.opacity) * b0 + self.opacity * self.blue
                    img[i][j] = [b1, g1, r1]

        return img
