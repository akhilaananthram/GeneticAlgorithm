#include "Point.h"
#include <stdlib.h>

//TODO define width and height somewhere
#define WIDTH 100
#define HEIGHT 100


Point::Point(int x, int y) :
x(x),
y(y)
{}

Point::Point() {
	x = rand() % WIDTH;
	y = rand() % HEIGHT;
}