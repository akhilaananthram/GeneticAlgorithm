#include <iostream>
#include "Threshold.h"
#include "Fitness.h"
#include "GeneticAlgorithm.h"
#include "HillStepper.h"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#define NUM_POLYGON_PROPERTIES 6
#define NUM_POPULATION_PROPERTIES 3
#define NUM_EVOLVE_PROPERTIES 4

using namespace rapidjson;
using namespace cv;

int WIDTH, HEIGHT;
struct Threshold t;

void parse_threshold(const char* threshold_file) {
	FILE* tFile = fopen(threshold_file, "rb");
	char buffer[65536];
	FileReadStream is(tFile, buffer, sizeof(buffer));
	Document d;
	d.ParseStream<FileReadStream>(is);
	Value& p = d["polygon"];
	double poly_props[NUM_POLYGON_PROPERTIES] = {
		p["red"].GetDouble(),
		p["blue"].GetDouble(),
		p["green"].GetDouble(),
		p["opacity"].GetDouble(),
		p["points"].GetDouble(),
		p["remove"].GetDouble()
	};
	for (int i = 1; i < NUM_POLYGON_PROPERTIES - 1; i++) {
		poly_props[i] += poly_props[i - 1];
	}
	t.polygon_properties = {
		(float)poly_props[0],
		(float)poly_props[1],
		(float)poly_props[2],
		(float)poly_props[3],
		(float)poly_props[4],
		(float)poly_props[5]
	};
	Value& pop = d["population"];
	t.population_properties = {
		(float)pop["mutate"].GetDouble(),
		(float)pop["modify"].GetDouble(),
		(float)pop["remove"].GetDouble()
	};
	Value& ev = d["evolve"];
	t.evolve_properties = {
		(float)ev["niche"].GetDouble(),
		(float)ev["mutate"].GetDouble(),
		(float)ev["elitism"].GetDouble(),
		(float)ev["random"].GetDouble()
	};
	fclose(tFile);
}

int main(int argc, const char * argv[]) {
	
	//parse_threshold("defaults.json");
    Mat original = imread("../../../images/mona2.jpg", CV_LOAD_IMAGE_COLOR);
    WIDTH = original.rows;
    HEIGHT = original.cols;
    
    Fitness f = Fitness(original);
    
    std::cout << "Hello, World!\n";
	//system("PAUSE");
    return 0;
}
