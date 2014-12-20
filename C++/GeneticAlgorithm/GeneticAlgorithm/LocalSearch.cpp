#include <stdlib.h>
#include "LocalSearch.h"

LocalSearch::LocalSearch(Mat original, Fitness f, int max_iterations) :
original(original),
fit(f),
max_iterations(max_iterations)
{};

Mat LocalSearch::draw(vector<Polygon> * plys) {
  Mat img = Mat::zeros(original.rows, original.cols, CV_64FC3);
  
  //add all the polygons to the image
  for (vector<Polygon>::iterator it=plys->begin(); it!=plys->end(); it++) {
      //create mask
      Mat mask = Mat::zeros(original.rows, original.cols, CV_64FC1);
      Polygon p = (* it);
      int num_points = (int) p.points.size();
      Point * polygon_points = new Point[num_points];
      int i = 0;
      for(vector<Point>::iterator ip=p.points.begin(); ip!=p.points.end(); ip++) {
          polygon_points[i] = (* ip);
      }
      const Point* ppt[1] = {polygon_points};
      
      const int npt = {10};
      fillPoly(mask, ppt, &npt, 1, Scalar(1), 8);
      delete polygon_points;
      
      //factor in opacity
      for (i=0; i<original.rows;i++) {
          for (int j=0; j<original.cols; j++) {
              if (mask.at<bool>(i,j)){
                  int b0 = img.at<int>(i,j,0);
                  int g0 = img.at<int>(i,j,1);
                  int r0 = img.at<int>(i,j,2);
                  img.at<int>(i,j,0) = (1.0 - p.opacity) * b0 + p.opacity * p.blue;
                  img.at<int>(i,j,1) = (1.0 - p.opacity) * g0 + p.opacity * p.green;
                  img.at<int>(i,j,2) = (1.0 - p.opacity) * r0 + p.opacity * p.red;
              }
          }
      }
  }
    
  return img;
}

float LocalSearch::fitness(vector<Polygon>* plys) {
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
