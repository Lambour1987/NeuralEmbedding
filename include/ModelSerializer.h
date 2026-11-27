//27-11-26


#pragma once

#include "Sequential.h"

#include <string>

class ModelSerializer {
public:
    void save(
        const Sequential& network,
        const std::string& filename
    ) const;

    void load(
        Sequential& network,
        const std::string& filename
    ) const;
};
