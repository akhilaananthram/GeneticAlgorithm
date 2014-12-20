#include "LocalSearch.h"
class HillStepper : public LocalSearch {
public:
	HillStepper(Mat original, Fitness f, int max_iterations, bool sim_an);

};