#include "Point.h"
#include <vector>

class Polygon {
public:
	Polygon(int start_points = 3, int r = -1, int b = -1, int g = -1, float o = -1.0);
	void mutate();

private:
	std::vector<Point> points;
	int red, blue, green;
	float opacity;
	//these functions add/remove a random point
	void add_vertex();
	void remove_vertex();
	//these functions randomize their respective values
	void change_red();
	void change_blue();
	void change_green();
	void change_opacity();
};