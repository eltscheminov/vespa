// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#pragma once

#include <stdio.h>

namespace slobrok {
namespace api {

class BackOff
{
private:
    double _time;
    double _since_last_warn;
    size_t _nextwarn_idx;

public:
    BackOff();
    void reset();
    double get();
    bool shouldWarn();
};

} // namespace api
} // namespace slobrok

