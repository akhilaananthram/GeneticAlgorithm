#include "GeneticAlgorithm.h"

GeneticAlgorithm::GeneticAlgorithm(Mat original, Fitness f, int max_iterations, int population_size, int parents, float niche_penalty) :
LocalSearch(original, f, max_iterations),
population_size(population_size),
num_parents(parents),
niche_penalty(niche_penalty)
{};

vector<Polygon>* GeneticAlgorithm::reservoir_sampling(vector<Polygon>* parent, int num_genes) {
    
}

vector<Polygon>* GeneticAlgorithm::create_child(vector<Polygon>** population, float pop_thresholds[]) {
    
}

vector<Polygon>** GeneticAlgorithm::evolve(vector<Polygon>** population, float pop_fitness[]) {
    
}
