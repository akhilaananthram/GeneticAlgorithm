#include <stdlib.h>
#include "LocalSearch.h"

LocalSearch::LocalSearch(Mat original, Fitness f, int max_iterations):
original(original);
fit(f);
max_iterations(max_iterations);
{}

Mat LocalSearch::draw(Polygon[] plys) {
  Mat img = Mat::zeros(original.rows, original.cols, CV_32F);
}

float LocalSearch::fitness(Polygon[] plys) {
  Mat img = draw(plys);
  float f = fit.score(img);

  return f;
}

Polygon[] randomPerson() {
  int num_polys = rand() % 10 + 1;
  Polygon person[num_polys];
  for (int i=0; i<num_polys; i++) {
    int num_points = rand() % 5 + 3; 
    person[i] = Polygon(num_points);
  }

  return person;
}
