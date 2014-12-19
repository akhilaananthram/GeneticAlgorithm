#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;

#define EUC 1

class Fitness {
public:
	Fitness(Mat original, int type = EUC, int sample = 1);
	float score(Mat img);

private:
  float euclidean(Mat img);
  Mat original;
  int type;
  int sample;
};
