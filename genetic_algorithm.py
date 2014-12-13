import cv2
from multiprocessing import Pool
import numpy as np
import random
import argparse
import copy
import math
import time
import sys

class Polygon(object):
    def __init__(self, xbound, ybound, points=None, red = None, blue = None, green = None, opacity = None):
        self.points = points
        if points is None:
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
            self.order_vertices()

        self.xbound = xbound
        self.ybound = ybound

        self.red = red
        if red is None:
            self.red =   int(random.random() * 256)

        self.blue = blue
        if blue is None:
            self.blue =  int(random.random() * 256)
        
        self.green = green
        if green is None:
            self.green = int(random.random() * 256)

        self.opacity = opacity
        if opacity is None:
            self.opacity = random.random()

    def add_vertex(self):
        x = int(random.random() * self.xbound)
        y = int(random.random() * self.ybound)
        self.points.append([x,y])
        self.order_vertices()

    def order_vertices(self):
        #calculate center point
        xc = 0.0
        yc = 0.0
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
        self.red = int(random.random() * 256)

    def change_green(self):
        self.green = int(random.random() * 256)

    def change_blue(self):
        self.blue = int(random.random() * 256)

    def mutate(self):
        mutation = int(random.random() * 150)
        if(mutation < 25):
            self.change_opacity()
        elif(mutation < 50):
            self.change_red()
        elif(mutation < 75):
            self.change_green()
        elif (mutation < 90):
            self.change_blue()
        else:
            if(len(self.points)> 3):
                if(int(random.random() * 2) == 0):
                    self.add_vertex()
                else:
                    self.remove_vertex()
            else:
                self.add_vertex()

    def __str__(self):
        poly = {
            "points" : self.points,
            "xbound" : self.xbound,
            "ybound" : self.ybound,
            "red" : self.red,
            "blue" : self.blue,
            "green" : self.green,
            "opacitiy" : self.opacity
            }

        return json.dumps(poly)

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

        self.type = type
        if type == "euc":
            self.num_proc = 3
            self.pool = Pool(self.num_proc)
            self.wstarts = [int((self.w / self.num_proc) * i) for i in xrange(self.num_proc)]
            self.wends = [int((self.w / self.num_proc) * (i + 1)) for i in xrange(self.num_proc)]
        elif type == "feat":
            self.detector = cv2.ORB()
            self.kp, self.desc = self.detector.detectAndCompute(self.original, None)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def euclidean(self, img):
        '''assumes img and self.original have the same size'''
        #start = time.time()
        distance = 0.0

        #pass to pool
        results = self.pool.map(euclidean_helper, [(img, self.original, self.wstarts[i], self.wends[i]) for i in xrange(self.num_proc)])

        for r in results:
          distance += r

        distance = distance / (self.w * self.h)

        #end = time.time()
        #print end - start
        return distance

    def feature_matching(self, img):
        #start = time.time()
        kp, desc = self.detector.detectAndCompute(img, None)

        matches = self.matcher.match(self.desc, desc)

        distance = 0.0

        for m in matches:
            distance += m.distance

        if len(matches) != 0:
            distance = distance / len(matches)

        #end = time.time()
        #print end - start

        if distance == 0:
            return sys.maxint

        return 1 / distance
    
    def score(self, img):
        if (self.type == "euc"):
            return self.euclidean(img)
        elif (self.type == "feat"):
            return self.feature_matching(img)

class Thresholds(object):
    def __init__(self, threshold_file, pop_size):
        #default initiation
        #POLYGON MUTATION
        self.opacity = .2
        self.red = .2
        self.green = .2
        self.blue = .2
        self.points = .2
        self.remove_point = .5
        #POPULATION MUTATION
        self.pop_add = .3
        self.pop_remove = .3
        self.pop_mutate_poly = .4
        #EVOLVE
        self.mutation = .3
        self.cross_breed = .7
        self.elitism = None
        self.add_random = 0.005

        if threshold_file is not None:
            #use json to read in dictionary
            thresholds = json.load(threshold_file)

            #POLYGON MUTATION
            polygon = thresholds.get("polygon", {})
            self.opacity = polygon.get("opacity", self.opacity)
            self.red = polygon.get("red", self.red)
            self.green = polygon.get("green", self.green)
            self.blue = polygon.get("blue", self.blue)
            self.points = polygon.get("points", self.points)
            self.remove_point = polygon.get("remove_point", self.remove_point)

            total = self.opacity + self.red + self.green + self.blue + self.points + self.remove_point
            self.opacity = self.opacity / total
            self.red = self.red / total
            self.green = self.green / total
            self.blue = self.blue / total
            self.points = self.points / total
            self.remove_point = self.remove_point / total

            #POPULATION MUTATION
            population = thresholds.get("population", {})
            self.pop_add = population.get("add", self.pop_add)
            self.pop_remove = population.get("remove", self.pop_remove)
            self.pop_mutate_poly = population.get("mutate", self.pop_mutate_poly)

            total = self.pop_add + self.pop_remove + self.pop_mutate_poly
            self.pop_add = self.pop_add / total
            self.pop_remove = self.pop_remove / total
            self.pop_mutate_poly = self.pop_mutate_poly / total

            #EVOLVE
            evolve = thresholds.get("evolve", {})
            self.mutation = evolve.get("mutate", self.mutation)
            self.cross_breed = evolve.get("cross_breed", self.cross_breed)
            
            total = self.mutation + self.cross_breed
            self.mutation = self.mutation / total
            self.cross_breed = self.cross_breed / total
            
            self.add_random = evolve.get("add_random", self.add_random)
            elitism = evolve.get("elitism", self.elitism)
            #not valid elitism
            if elitism <= 0 or elitism is None or elitism > pop_size:
                self.elitism = None
            #already proportion
            elif elitism < 1:
                self.elitism = elitism
            #make into proportion
            else:
                self.elitism = elitism / pop_size

class Driver(object):
    def __init__(self, args, t):
        self.original = cv2.imread(args.path)
        self.fit = Fitness(self.original, args.fitness)
        self.iterations = args.iterations
        self.w = self.original.shape[1]
        self.h = self.original.shape[0]
        self.max_poly = 1

        self.t = t

    def draw(self, polygons):
        img = np.zeros(self.original.shape, dtype=np.uint8)
        
        for p in polygons:
            poly = np.array(p.points, np.int32)
            h, w, _ = img.shape
            mask = np.zeros((h,w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 1)

            #opacity
            for i in xrange(w):
                for j in xrange(h):
                    #polygon exists there
                    if mask[j][i]:
                        b0, g0, r0 = img[j][i]
                        r1 = np.uint8((1.0 - p.opacity) * r0 + p.opacity * p.red)
                        g1 = np.uint8((1.0 - p.opacity) * g0 + p.opacity * p.green)
                        b1 = np.uint8((1.0 - p.opacity) * b0 + p.opacity * p.blue)
                        img[j][i] = [r1, g1, b1]

        cv2.imwrite("temp.png", img)
        return img

    def mutate(self, plys):
        val = int(random.random() * 100)
        if(val < 30):
            to_mutate = int(random.random() * len(plys))
            plys[to_mutate].mutate()
        else:
            plys.append(Polygon(self.w, self.h))
            if len(plys) > self.max_poly:
                self.max_poly = len(ply)
        """else:
            to_remove = int(random.random() * len(plys))
            plys.pop(to_remove)"""
    
    def random_person(self):
        person = []
        num_polys = random.randrange(1,self.max_poly + 1)
        for i in xrange(num_polys):
            poly = Polygon(self.w, self.h)
            num_points = random.randrange(3, 7)
            for j in xrange(3, num_points):
                poly.add_vertex()

    def fitness(self, plys):
        return self.fit.score(self.draw(plys))

    def run(self):
        return None

class HillSteppingDriver(Driver):
    def __init__(self, args, t):
        Driver.__init__(self, args, t)

    def step(self, polygons, fit):
        newpolygons = copy.deepcopy(polygons)
        self.mutate(newpolygons)

        while(self.fitness(newpolygons) >= fit):
            newpolygons = copy.deepcopy(polygons)
            self.mutate(newpolygons)
        return newpolygons

    def run(self):
        polygons = self.random_person()
        iterations = 0
        while True:
            if self.iterations != None and self.iterations == iterations:
                return polygons

            fit = self.fitness(polygons)
            if(fit < 1):
                return polygons
            polygons = self.step(polygons, fit)
            #print "leave step"
            iterations += 1
            print iterations

class GeneticAlgorithmDriver(Driver):
    def __init__(self, args, t):
        Driver.__init__(self, args, t)
        self.pop_size = args.population

        if args.parents < self.pop_size:
            self.num_parents = args.parents
        else:
            self.num_parents = 2

    def cross_breed(parents):
        num_from_parent = 0
        for p in parents:
            num_from_parent += len(p)

        #average length / num_parents
        num_from_parent = size / (2 * len(parents))

        child = []
        for p in parents:
            child = child + p[:num_from_parent]
        return child

    def evolve(self, population, pop_fitness):
        children = []
        #TODO: NICHE PENALTY

        thresholds = [(1.0 - i / total) for i in pop_fitness]
        for i in xrange(1, len(thresholds)):
            thresholds[i] += thresholds[i - 1]

        for i in xrange(self.pop_size):
            if random.random() < .5:
                #cross breed
                parents = []
                parent_indices = set()
                for i in xrange(self.num_parents):
                    parent = None
                    while parent == None:
                        r = random.random()
                        p = 0
                        while r > thresholds[p]:
                            p += 1
                        if p not in parent_indices:
                            parent = thresholds[p]

                for i in parent_indices:
                    parents.append(population[i])

                child = self.cross_breed(parents)
            else:
                #mutate
                r = random.random()
                p = 0
                while r > thresholds[p]:
                    p += 1

                parent = population[p]
                child = copy.deepcopy(parent)
                self.mutate(child)

            children.append(child)

        if self.elitism is not None:
            #get (self.elitism * self.pop_size) best parents
            population, probabilities = zip(*sorted(zip(population, pop_fitness), key=lambda p:p[1], reverse=True))

            num_parents = int(self.elitism * self.pop_size)
            lasting_parents = population[:num_parents]

            #get ((1 - self.elitism) * self.pop_size) best children
            fitness = [self.fitness(c) for c in children]
            children, fitness = zip(*sorted(zip(children, fitness), key=lambda c:c[1], reverse=True))
            lasting_children = children[:(self.pop_size - num_parents)]

            children = lasting_parents + lasting_children

        if random.random() < .005:
            #pick a random child to pop and then replace with random
            index = random.randrange(0, len(children))
            children[index] = self.random_person()

        return children

    def run(self):
        #generate parents
        population = [self.random_person() for i in xrange(self.pop_size)]
        iterations = 0
        while True:
            if self.iterations != None and self.iterations == iterations:
                return polygons

            fit = [self.fitness(p) for p in population]
            if (min(fit) < 1):
                index = np.argmin(np.array(fit))
                return population[index]

            total = 0
            for i in fit:
                total += i

            population = self.evolve(population, fit)
            #print "leave crossbreed""
            iterations += 1
            print iterations

def parse_args():
    parser = argparse.ArgumentParser(description="Test Genetic Algorithms")
    parser.add_argument("--algorithm", dest="algo", type=str, choices=["hill", "genetic"],
        help="Type of algorithm to use", default="genetic")
    parser.add_argument("--fitness", dest="fitness", default="euc", choices=["euc", "feat"],
        help="Type of fitness function to use.")

    parser.add_argument("--path", dest="path", type=str, help="Path to image. REQUIRED", required=True)
    parser.add_argument("--dest", dest="dest", type=str, help="Path for destination image", default = None)

    parser.add_argument("--iterations", dest="iterations", type=int, default=None, help="Number of iterations to do.")
    parser.add_argument("--population", dest="population", type=int, default=5, help="Population size.")
    parser.add_argument("--parents", dest="parents", type=int, default=2, help="Number of parents for cross breeding.")
    parser.add_argument("--elitism", dest="elitism", type=float, default=None, help="Percent of parents to survive.")

    parser.add_argument("--thresholds", dest="thresholds", type=str, default=None, help="Path to json file of thresholds.")

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()

    t = Threshold(args.thresholds, args.population)

    if args.algo == "genetic":
        d = GeneticAlgorithmDriver(args, t)
    else:
        d = HillSteppingDriver(args, t)

    polygons = d.run()

    #save image
    dest = args.dest
    if args.dest is None:
        dest = args.path.split(".") + "_result.jpg"
    result = d.draw(polygons)
    cv2.imwrite(dest, result)

    #save polygons
    with file("polygons.poly", "w") as f:
        for poly in polygons:
            s = poly.__str__()
            f.write(s + "\n")
