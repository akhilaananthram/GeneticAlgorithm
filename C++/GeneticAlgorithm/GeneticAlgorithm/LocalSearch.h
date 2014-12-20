#ifndef __LocalSearch__LocalSearch__
#define __LocalSearch__LocalSearch__

#include <opencv2/opencv.hpp>
#include "Fitness.h"
#include "Polygon.h"

using namespace cv;
using namespace std;

class LocalSearch {
public:
  LocalSearch(Fitness f, int max_iterations);
  virtual std::vector<Polygon>* run() = 0;
  std::vector<Polygon>* randomPerson();
  float fitness(vector<Polygon>* plys);
  int max_iterations;
  void mutate(std::vector<Polygon>* polys);

private:
  Mat draw(vector<Polygon>* plys);
  Fitness fit;
};

#endif /* defined(__LocalSearch__LocalSearch__) */