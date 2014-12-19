#include <opencv2/opencv.hpp>
#include "Fitness.h"
#include "Polygon.h"

using namespace cv;

class LocalSearch {
public:
  LocalSearch(Mat original, Fitness f, int max_iterations);
  virtual void run();

private:
  Mat draw(Polygon[] plys);
  float fitness(Polygon[] plys);
  Polygon[] randomPerson();
  Mat original;
  Fitness fit;
  int max_iterations;
}
