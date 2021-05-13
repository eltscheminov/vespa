# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath("../../vespa/ann_bm"))
from vespa_ann_bm import DistanceMetric, HnswIndexParams, AnnBm

class Fixture_2d:
    def __init__(self):
        self.tensor = AnnBm(2, HnswIndexParams(16, 200, DistanceMetric.Euclidean, False))

    def set(self, lid, value):
       self.tensor.set_value(lid, value)

    def get(self, lid):
        return self.tensor.get_value(lid)

    def find(self, k, value):
        return self.tensor.find_top_k(k, value, k + 200, 1e300)

def test_2d_euclidean():
    f = Fixture_2d()
    f.set(0, [0, 0])
    f.set(1, [10, 10])
    assert f.get(0) == [0, 0]
    assert f.get(1) == [10, 10]
    top = f.find(10, [1, 1])
    print("top is ", top)
    assert [top[0][0], top[1][0]] == [0, 1]
    top2 = f.find(10, [9, 9])
    print("top2 is ", top2)
    assert [top2[0][0], top2[1][0]] == [1, 0]
