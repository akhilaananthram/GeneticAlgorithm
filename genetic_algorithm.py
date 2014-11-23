import cv2
import numpy as np
import random

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

    def remove_vertex(self):
        to_remove = int(random.random() * len(self.points))
        self.points.pop(to_remove)

    def change_opacity(self):
        self.opacity = random.random()

    def change_color(self):
        self.red =   int(random.random() * 256)
        self.green = int(random.random() * 256)
        self.blue =  int(random.random() * 256)

    def mutate(self):
        mutation = int(random.random() * 100)
        if(mutation < 5):
            change_opacity()
        elif(mutation < 15):
            change_color()
        elif(mutation < 40):
            if(len(self.points)> 3):
                if(int(random.random * 2) == 0):
                    add_vertex()
                else:
                    remove_vertex()
            else:
                add_vertex()

class Driver(object):
    def __init__(self, img):
        self.original = img

    def draw_on_image(self, polygon, img):
        poly = np.array(polygon.points, np.int32)
        w, h, _ = img.shape
        mask = np.zeros((w, h))
        cv2.fillPoly(mask, [poly], 1)

        #opacity
        for i in xrange(w):
            for j in xrange(h):
                #polygon exists there
                if mask[i][j]:
                    b0, g0, r0 = img[i][j]
                    r1 = (1.0 - polygon.opacity) * r0 + polygon.opacity * polygon.red
                    g1 = (1.0 - polygon.opacity) * g0 + polygon.opacity * polygon.green
                    b1 = (1.0 - polygon.opacity) * b0 + polygon.opacity * polygon.blue
                    img[i][j] = [b1, g1, r1]

        return img

    #resets canvas
    def clear_image(self, img):
        pass

    #takes a list of polysgons
    #draws them all to the same image
    #computers the numeric fitness
    #clears the image
    #returns the fitness score
    def fitness(self, plys):
        return 1
        #for p in plys:
        #    draw_on_image(p, img)

        #blank_image()

    def mutate(self, plys):
        if(int(random.random*3) ==0):
            to_mutate = int(random.random() * len(plys))
            plys[to_mutate].mutate()
        elif(int(random.random*3) ==1):
            plys.append(Polygon())#add bounds 
        else:
            to_remove = int(random.random() * len(plys))
            plys.pop(to_remove)

    def cross_breed(self, polygons):
        newpolygons = polygons
        mutate(newpolygons)
        while(fitness(newpolygons) <= fitness(polygons)):
            newpolygons = polygons
            mutate(newpolygons)
        return newpolygons

    def run(self):
        polygons = [Polygon()] #add bounds
        while True:
            if(fitness(polygons) > .9):
                return polygons
            polygons = cross_breed(polygons)




d = Driver(img) #make this the mona lisa
d.draw_on_image(d.run(), img) #set image