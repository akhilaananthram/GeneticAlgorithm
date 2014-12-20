#include "GeneticAlgorithm.h"
#include "Threshold.h"
#include <algorithm>

extern struct Threshold t;

GeneticAlgorithm::GeneticAlgorithm(Fitness f, int max_iterations, int population_size, int parents, float niche_penalty) :
LocalSearch(f, max_iterations),
population_size(population_size),
num_parents(parents),
niche_penalty(niche_penalty)
{};

vector<Polygon>* GeneticAlgorithm::reservoir_sampling(vector<Polygon>* parent, int num_genes) {
    vector<Polygon> * genes = new vector<Polygon>();
    for (int i=0; i<num_genes;i++) {
        genes->push_back(parent->at(i));
    }
    
    for (int i=num_genes; i<parent->size();i++) {
        int j = rand() % i;
        
        if (j < num_genes) {
            genes->at(j) = parent->at(i);
        }
    }
    
    return genes;
}

vector<Polygon>* GeneticAlgorithm::create_child(vector<Polygon>* population[], float pop_thresholds[]) {
    vector<Polygon> * child = new vector<Polygon>();
    
    set<int> * parent_indices = new set<int>();
    for (int i=0; i<num_parents; i++) {
        int parent = -1;
        while (parent == -1) {
            float r = ((float)rand()) / RAND_MAX;
            int p = 0;
            while (r > pop_thresholds[p]) {
                p++;
            }
            
            //p is not in the set
            if (parent_indices->find(p) == parent_indices->end()) {
                parent = p;
            }
        }
        parent_indices->insert(parent);
    }
    
    int num_from_parent = 0;
    for (set<int>::iterator it=parent_indices->begin(); it!=parent_indices->end(); it++) {
        vector<Polygon> * parent = population[(* it)];
        num_from_parent += parent->size();
    }
    
    num_from_parent /= num_parents;
    
    for (set<int>::iterator it=parent_indices->begin(); it!=parent_indices->end(); it++) {
        vector<Polygon> * parent = population[(* it)];
        int num_genes = min((int)parent->size(), (int)num_from_parent);
        vector<Polygon>* genes = reservoir_sampling(parent, num_genes);
        //TODO: deep copy genes to child
    }
    
    float m = ((float)rand()) / RAND_MAX;
    if (m < t.evolve_properties.mutate) {
        mutate(child);
    }
    
    return child;
}

vector<Polygon>** GeneticAlgorithm::evolve(vector<Polygon>* population[], float pop_fitness[]) {
    vector<Polygon>** children = new vector<Polygon>*();
    
    return children;
}

std::vector<Polygon>* LocalSearch::run() {
    vector<Polygon> * best;
    
    return best;
}