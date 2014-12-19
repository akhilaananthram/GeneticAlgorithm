#include <stdlib.h>
#include "LocalSearch.h"

LocalSearch::LocalSearch(Mat original, Fitness f, int max_iterations) :
original(original),
fit(f),
max_iterations(max_iterations)
{};

Mat LocalSearch::draw(Polygon plys[]) {
  return Mat::zeros(original.rows, original.cols, CV_32F);
}

float LocalSearch::fitness(Polygon plys[]) {
  Mat img = draw(plys);
  float f = fit.score(img);
  return f;
}

std::vector<Polygon>* LocalSearch::randomPerson() {
  int num_polys = rand() % 10 + 1;
  std::vector<Polygon>* person = new std::vector<Polygon>(num_polys);
  for (int i=0; i<num_polys; i++) {
    int num_points = rand() % 5 + 3; 
    person->push_back(Polygon(num_points));
  }
  return person;
}
