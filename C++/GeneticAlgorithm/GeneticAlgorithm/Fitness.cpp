#include "Fitness.h"
#include <math.h>

Fitness::Fitness(Mat original, int type, int sample) :
original(original),
type(type),
sample(sample)
{};

float Fitness::score(Mat img) {
  switch(type){
    case EUC:
        return euclidean(img);
	default:
		throw "Unknown fitness type!";
  }
}

float Fitness::euclidean(Mat img) {
  float distance = 0.0;
  for(int i=0; i<img.rows; i+=sample){
    for(int j=0; j<img.cols; j+=sample){
      Vec3b source = original.at<Vec3b>(i,j);
      Vec3b data = img.at<Vec3b>(i,j);

      distance += sqrt(source[0] * data[0] + source[1] * data[1] + source[2] * data[2]);
    }
  }

  distance /= (img.rows * img.cols);

  return distance;
}
