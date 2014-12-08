import cv2
from multiprocessing import Pool
import numpy as np
import random
import argparse
import copy
import math
import time

class Polygon(object):
    def __init__(self, xbound, ybound):
        self.points = []

        x = int(random.random() * xbound)
        y = int(random.random() * ybound)
        self.points.append([x,y])
        x = int(random.random() * xbound)
        y = int(random.random() * ybound)
        self.points.append([x,y])
        x = int(random.random() * xbound)
        y = int(random.random() * ybound)
        self.points.append([x,y])

        self.xbound = xbound
        self.ybound = ybound
        self.red =   int(random.random() * 256)
        self.blue =  int(random.random() * 256)
        self.green = int(random.random() * 256)
        self.opacity = random.random()

    def add_vertex(self):
        x = int(random.random() * self.xbound)
        y = int(random.random() * self.ybound)
        self.points.append([x,y])

    def order_vertices(self):
        #calculate center point
        xc = 0
        yc = 0
        for x, y in self.points:
            xc += x
            yc += y

        xc = xc / len(self.points)
        yc = yc / len(self.points)

        #sort
        self.points = sorted(self.points, key=lambda p: math.atan2(p[1] - yc, p[0] - xc))

    def remove_vertex(self):
        to_remove = int(random.random() * len(self.points))
        self.points.pop(to_remove)

    def change_opacity(self):
        self.opacity = random.random()

    def change_red(self):
        self.red =   int(random.random() * 256)

    def change_green(self):
        self.green = int(random.random() * 256)

    def change_blue(self):
        self.blue =  int(random.random() * 256)

    def mutate(self):
        mutation = int(random.random() * 100)
        if(mutation < 25):
            self.change_opacity()
        elif(mutation < 50):
            self.change_red()
        elif(mutation < 75):
            self.change_green()
        else:
            self.change_blue()
            """
        elif(mutation < 40):
            if(len(self.points)> 3):
                if(int(random.random * 2) == 0):
                    self.add_vertex()
                else:
                    self.remove_vertex()
            else:
                self.add_vertex()"""

def euclidean_helper(args):
    img, original, wstart, wend = args
    '''assumes img and self.original have the same size'''
    #set up
    distance = 0.0

    for i in xrange(wstart, wend):
        for j in xrange(original.shape[0]):
            distance += np.linalg.norm(img[j][i] - original[j][i])

    return distance

class Fitness(object):
    def __init__(self, original, type="euc"):
        self.original = original
        self.w = original.shape[1]
        self.h = original.shape[0]

        self.detector = cv2.ORB()
        self.kp, self.desc = self.detector.detectAndCompute(self.original, None)
        self.type = type
        self.num_proc = 3
        self.pool = Pool(self.num_proc)
        self.wstarts = [int((self.w / self.num_proc) * i) for i in xrange(self.num_proc)]
        self.wends = [int((self.w / self.num_proc) * (i + 1)) for i in xrange(self.num_proc)]

    def euclidean(self, img):
        '''assumes img and self.original have the same size'''
        distance = 0.0

        #pass to pool
        results = self.pool.map(euclidean_helper, [(img, self.original, self.wstarts[i], self.wends[i]) for i in xrange(self.num_proc)])

        for r in results:
          distance += r

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
    
    def score(self, img):
        if (self.type == "euc"):
            return self.euclidean(img)
        elif (self.type == "feat"):
            return self.feature_matching(img)

class Driver(object):
    def __init__(self, img, iterations=None):
        self.original = img
        self.fit = Fitness(self.original)
        self.iterations = iterations
        self.w = self.original.shape[1]
        self.h = self.original.shape[0]

    def draw(self, polygons):
        img = np.zeros(self.original.shape)
        
        for p in polygons:
            poly = np.array(p.points, np.int32)
            h, w, _ = img.shape
            mask = np.zeros((h,w))
            cv2.fillPoly(mask, [poly], 1)

            #opacity
            for i in xrange(w):
                for j in xrange(h):
                    #polygon exists there
                    if mask[j][i]:
                        b0, g0, r0 = img[j][i]
                        r1 = (1.0 - p.opacity) * r0 + p.opacity * p.red
                        g1 = (1.0 - p.opacity) * g0 + p.opacity * p.green
                        b1 = (1.0 - p.opacity) * b0 + p.opacity * p.blue
                        img[j][i] = [r1, g1, b1]

        cv2.imwrite("temp.png", img)
        return img

    def fitness(self, plys):
        return self.fit.score(self.draw(plys))

    def mutate(self, plys):
        val = int(random.random() * 100)
        if(val < 30):
            to_mutate = int(random.random() * len(plys))
            plys[to_mutate].mutate()
        else:
            plys.append(Polygon(self.w, self.h))#add bounds 
        """else:
            to_remove = int(random.random() * len(plys))
            plys.pop(to_remove)"""

    def cross_breed(self, polygons):
        newpolygons = copy.deepcopy(polygons)
        self.mutate(newpolygons)
        fit = self.fitness(polygons)

        while(self.fitness(newpolygons) >= fit):
            newpolygons = copy.deepcopy(polygons)
            self.mutate(newpolygons)
            print "fitness loop: " + str(fit)
        return newpolygons

    def run(self):
        polygons = [Polygon(self.w, self.h)] #add bounds
        iterations = 0
        while True:
            if self.iterations != None and self.iterations == iterations:
                return polygons

            if(self.fitness(polygons) < 1):
                return polygons
            polygons = self.cross_breed(polygons)
            print "leave crossbreed"
            iterations += 1
            print iterations

def parse_args():
    parser = argparse.ArgumentParser(description="Test Genetic Algorithms")
    parser.add_argument("--path", dest="path", type=str, help="Path to image. REQUIRED", required=True)

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()

    img = cv2.imread(args.path)
    d = Driver(img)

    polygons = d.run()
    result = d.draw(polygons)
    cv2.imwrite("result.jpg", result)
