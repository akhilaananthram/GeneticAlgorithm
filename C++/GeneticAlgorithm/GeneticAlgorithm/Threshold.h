struct Threshold {
	struct polygon_properties {
		float red, blue, green, opacity, points, remove;
	} polygon_properties;
	struct population_properties {
		float mutate, modify, remove;
	} population_properties;
	struct evolve_properties {
		float niche, mutate, elitism, random;
	} evolve_properties;
};