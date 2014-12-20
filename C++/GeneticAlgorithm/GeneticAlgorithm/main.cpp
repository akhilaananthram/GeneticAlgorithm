#include <iostream>
#include "Polygon.h"
#include "Threshold.h"
#include "rapidjson\document.h"
#include "rapidjson\filereadstream.h"
#include "rapidjson\stringbuffer.h"
#include "rapidjson\writer.h"

#define NUM_POLYGON_PROPERTIES 6
#define NUM_POPULATION_PROPERTIES 3
#define NUM_EVOLVE_PROPERTIES 4

using namespace rapidjson;

int WIDTH = 100, HEIGHT = 100;
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
		poly_props[0], 
		poly_props[1], 
		poly_props[2],
		poly_props[3],
		poly_props[4],
		poly_props[5]
	};
	Value& pop = d["population"];
	t.population_properties = {
		pop["mutate"].GetDouble(),
		pop["modify"].GetDouble(),
		pop["remove"].GetDouble()
	};
	Value& ev = d["evolve"];
	t.evolve_properties = {
		ev["niche"].GetDouble(),
		ev["mutate"].GetDouble(),
		ev["elitism"].GetDouble(),
		ev["random"].GetDouble()
	};
	fclose(tFile);
}

int main(int argc, const char * argv[]) {
	
	parse_threshold("defaults.json");
	Polygon p = Polygon();
    std::cout << "Hello, World!\n";
	system("PAUSE");
    return 0;
}
