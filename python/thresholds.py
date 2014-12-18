import json

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
        self.pop_mutate_poly = .4
        self.modify_list = .6
        self.remove = 0.05
        #EVOLVE
        self.niche = 0
        self.mutation = .1
        self.elitism = None
        self.add_random = 0

        if threshold_file is not None:
            #use json to read in dictionary
            with file(threshold_file) as f:
                thresholds = json.load(f)

                #POLYGON MUTATION
                polygon = thresholds.get("polygon", {})
                self.opacity = polygon.get("opacity", self.opacity)
                self.red = polygon.get("red", self.red)
                self.green = polygon.get("green", self.green)
                self.blue = polygon.get("blue", self.blue)
                self.points = polygon.get("points", self.points)
                self.remove_point = polygon.get("remove", self.remove_point)

                #POPULATION MUTATION
                population = thresholds.get("population", {})
                self.pop_mutate_poly = population.get("mutate", self.pop_mutate_poly)
                self.modify_list = population.get("modify", self.modify_list)
                remove = population.get("remove", self.remove)
                if 0 <= remove <= 1:
                    self.remove = remove

                #EVOLVE
                evolve = thresholds.get("evolve", {})

                self.niche = abs(evolve.get("niche", self.niche))
                mutation = evolve.get("mutate", self.mutation)
                if 0 <= mutation <= 1:
                    self.mutation = mutation
                
                self.add_random = evolve.get("random", self.add_random)

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

        total = self.opacity + self.red + self.green + self.blue + self.points + self.remove_point
        self.opacity = self.opacity / total
        self.red = self.red / total
        self.green = self.green / total
        self.blue = self.blue / total
        self.points = self.points / total
        self.remove_point = self.remove_point / total

        total = self.pop_mutate_poly + self.modify_list
        self.pop_mutate_poly = self.pop_mutate_poly / total
        self.modify_list = self.modify_list / total

        self.polygon = [self.opacity, self.red, self.green, self.blue, self.remove_point]
        for i in xrange(1, len(self.polygon)):
            self.polygon[i] += self.polygon[i - 1]

