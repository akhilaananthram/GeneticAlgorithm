#include <opencv2/opencv.hpp>
#include "Fitness.h"
#include "Polygon.h"

using namespace cv;
using namespace std;

class LocalSearch {
public:
  LocalSearch(Mat original, Fitness f, int max_iterations);
  virtual void run() = 0;

private:
  Mat draw(vector<Polygon>* plys);
  float fitness(vector<Polygon>* plys);
  std::vector<Polygon>* randomPerson();
  Mat original;
  Fitness fit;
  int max_iterations;
};
