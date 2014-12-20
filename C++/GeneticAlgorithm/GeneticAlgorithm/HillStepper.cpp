#include "HillStepper.h"

HillStepper::HillStepper(Mat original, Fitness f, int max_iterations, bool sim_an) :
LocalSearch(original, f, max_iterations),
sim_an(sim_an)
{};

std::vector<Polygon>* HillStepper::run() {
	std::vector<Polygon>* polys = randomPerson();
	iters = 1;
	while (true) {
		if (max_iterations > 0 && iters == max_iterations) {
			return polys;
		}
		float fit = fitness(polys);
		if (fit < 1) {
			return polys;
		}
		polys = step(polys, fit);
		printf("Iteration %d\n", iters); 
		iters++;
	}
}

std::vector<Polygon>* HillStepper::step(std::vector<Polygon>* polys, float fit) {
	mutate(polys);

	if (sim_an && ((float)rand()) / RAND_MAX < (.5 / iters)) {
		while (fitness(polys) <= fit) {
			mutate(polys);
		}
	} else {
		while (fitness(polys) >= fit) {
			mutate(polys);
		}
	}
	return polys;
}