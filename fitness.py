import cv2
import numpy as np

class Fitness(object):
    def __init__(self, original):
        self.original = original
        self.w, self.h, _ = original.shape
        self.detector = cv2.ORB()
        self.kp, self.desc = self.detector.detectAndCompute(self.original, None)

    def euclidean(self, img):
        '''assumes img and self.original have the same size'''

        distance = 0.0
        for i in xrange(self.w):
            for j in xrange(self.h):
                distance += np.linalg.norm(img[i][j] - self.original[i][j])

        distance = distance / (self.w * self.h)
        return distance

    def feature_matching(self, img):
        kp, desc = self.detector.detectAndCompute(img, None)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = matcher.match(self.desc, desc)

        distance = 0.0

        for m in matches:
            distance += m.distance

        distance = distance / len(matches)
        return distance
