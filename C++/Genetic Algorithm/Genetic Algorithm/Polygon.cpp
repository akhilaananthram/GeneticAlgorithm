#include "Polygon.h"

Polygon::Polygon(int start_points, int r, int b, int g, float o):
red(r),
blue(b),
green(g),
opacity(o)
{
		Point center = Point(0, 0);
		for (int i = 0; i < start_points; i++) {
			Point p = Point();
			points.push_back(p);
			center.x += p.x;
			center.y += p.y;
		}
		center.x /= points.size();
		center.y /= points.size();
	
}