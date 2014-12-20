#ifndef __GeneticAlgorithm__GeneticAlgorithm__
#define __GeneticAlgorithm__GeneticAlgorithm__

#include <stdio.h>
#include "LocalSearch.h"

using namespace std;

class GeneticAlgorithm : public LocalSearch {
public:
    GeneticAlgorithm(Fitness f, int max_iterations, int population_size, int parents, float niche_penalty);
    vector<Polygon>* run();

private:
    vector<Polygon>* reservoir_sampling(vector<Polygon>* parent, int num_genes);
    vector<Polygon>* create_child(vector<vector<Polygon> *>* population, float pop_thresholds[]);
    vector<vector<Polygon> *>* evolve(vector<vector<Polygon> *>* population, float pop_fitness[]);
    int population_size;
    int num_parents;
    float niche_penalty;
};

#endif /* defined(__GeneticAlgorithm__GeneticAlgorithm__) */
