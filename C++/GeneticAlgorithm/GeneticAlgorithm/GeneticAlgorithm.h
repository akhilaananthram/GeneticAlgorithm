#ifndef __GeneticAlgorithm__GeneticAlgorithm__
#define __GeneticAlgorithm__GeneticAlgorithm__

#include <stdio.h>
#include "LocalSearch.h"
#include "Person.h"

using namespace std;

class GeneticAlgorithm : public LocalSearch {
public:
    GeneticAlgorithm(Fitness f, int max_iterations, int population_size, int parents, float niche_penalty);
    vector<Polygon>* run();

private:
    vector<Polygon>* reservoir_sampling(vector<Polygon>* parent, int num_genes);
    vector<Polygon>* create_child(vector<Person *>* population, float pop_thresholds[]);
    vector<Person *>* evolve(vector<Person *>* population);
    const int population_size;
    int num_parents;
    float niche_penalty;
};

#endif /* defined(__GeneticAlgorithm__GeneticAlgorithm__) */
