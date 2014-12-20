//
//  Person.h
//  GeneticAlgorithm
//
//  Created by Akhila Ananthram on 12/20/14.
//  Copyright (c) 2014 Akhila Ananthram. All rights reserved.
//

#include "Polygon.h"

#ifndef GeneticAlgorithm_Person_h
#define GeneticAlgorithm_Person_h

struct Person {
    vector<Polygon> * plys;
    float fit;
    bool operator()(Person const &a, Person const &b) {
        return a.fit < b.fit;
    }
};

#endif
