#include "Polygon.h"
#include <algorithm>
#include <math.h>

extern int WIDTH;
extern int HEIGHT;
/*
#define WIDTH 100
#define HEIGHT 100
*/
//todo figure out where these should go

using namespace cv;
using namespace std;

class SortHelper {
public:
	Point center;
	bool operator() (Point p, Point q) { return atan2(p.y - center.y, p.x - center.x) < atan2(q.y - center.y, q.x - center.x); };
};
Polygon::Polygon(int start_points)
{
	center = Point(0, 0);
	for (int i = 0; i < start_points; i++) {
		Point p = Point(rand() % WIDTH, rand() % HEIGHT);
		points.push_back(p);
		center.x += p.x;
		center.y += p.y;
	}
	int sz = points.size();
	center.x /= sz;
	center.y /= sz;
	SortHelper s;
	s.center = center;
	std::sort(points.begin(), points.end(), s);
	red = rand() % 255;
	blue = rand() % 255;
	green = rand() % 255;
	opacity = ((float)rand()) / RAND_MAX;
	/*
	float* arctans = new float[start_points];
	for (int i = 0; i < sz; i++) {
		Point p = points.at(i);
		arctans[i] = atan2(p.y - center.y, p.x - center.x);
	}
	*/
}

void Polygon::change_red() {
	red = rand() % 255;
}
void Polygon::change_blue() {
	blue = rand() % 255;
}
void Polygon::change_green() {
	green = rand() % 255;
}
void Polygon::change_opacity() {
	opacity = ((float)rand()) / RAND_MAX;
}
void Polygon::remove_vertex() {
	int r = rand() % points.size();
	points.erase(points.begin() + r);
}
void Polygon::add_vertex() {
	Point p = Point(rand() % WIDTH, rand() % HEIGHT);
	center.x *= points.size();
	center.y *= points.size();
	center.x += p.x;
	center.y += p.y;
	center.x /= points.size() + 1;
	center.y /= points.size() + 1;
	SortHelper s;
	s.center = center;
	int imin = 0, imid, imax = points.size();
	while (imin < imax) {
		imid = (imin + imax) / 2;
		bool b = s.operator()(p, points.at(imid));
		if (b) {
			imin = imid + 1;
		}
		else {
			imax = imid;
		}
	}
	if (imax == points.size()) {
		points.push_back(p);
	}
	else {
		points.insert(points.begin() + imax + 1, p);
	}

}