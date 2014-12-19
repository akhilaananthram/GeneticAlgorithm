#include "Point.h"
#include <vector>

class Polygon {
public:
	Polygon(int start_points = 3);
	void mutate();
	std::vector<Point> points;
	int red, blue, green;
	float opacity;

private:
	Point center;
	//these functions add/remove a random point
	void add_vertex();
	void remove_vertex();
	//these functions randomize their respective values
	void change_red();
	void change_blue();
	void change_green();
	void change_opacity();
};