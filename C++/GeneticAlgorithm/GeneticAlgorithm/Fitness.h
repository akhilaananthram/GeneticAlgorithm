#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;

class Fitness {
public:
  Fitness(Mat original, String type="euc", int sample=1):
  float score(Mat img);

private:
  float euclidean(Mat img);
  Mat original;
  String type;
  int sample;
};
