# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
import json
import unittest
from ._load import (BaseLoad, RollingStock, Locomotive, TwoAxleWagon,
                    ThreeAxleWagon, BogieWagon, JacobsWagon, Train)

__all__ = ["PacrilJSONEncoder", "PacrilJSONDecoder"]


class PacrilJSONEncoder(json.JSONEncoder):
    """Encode pacril objects to JSON objects, see json.dump/dumps

    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, BaseLoad) or isinstance(obj, RollingStock):
            return obj.todict()
        return super(PacrilJSONEncoder, self).default(obj)


class PacrilJSONDecoder(json.JSONDecoder):
    """Encode JSON objects to pacril objects, see json.load/loads

    """
    def __init__(self, *args, **kwargs):
        super(PacrilJSONDecoder, self).__init__(
            *args, object_hook=self.object_hook, **kwargs)

    def object_hook(self, obj):
        if "pacrilcls" not in obj:
            return obj
        pacrilcls = obj["pacrilcls"]
        if pacrilcls == "Load":
            return Load(obj["xp"], obj["p"])
        elif pacrilcls == "Locomotive":
            return Locomotive(obj["xp"], obj["p"])
        elif pacrilcls == "TwoAxleWagon":
            return TwoAxleWagon(obj["p"], obj["a"], obj["b"], obj["pempty"])
        elif pacrilcls == "ThreeAxleWagon":
            return ThreeAxleWagon(obj["p"], obj["a"], obj["b"], obj["pempty"])
        elif pacrilcls == "BogieWagon":
            return BogieWagon(
                obj["p"], obj["a"], obj["b"], obj["c"], obj["pempty"])
        elif pacrilcls == "JacobsWagon":
            return JacobsWagon(
                obj["p"], obj["a"], obj["b"], obj["c"], obj["pempty"])
        elif pacrilcls == "Train":
            return Train(obj["locomotive"], obj["wagons"])
        elif pacrilcls == "RollingStock":
            return RollingStock(
                obj["locomotives"], obj["wagons"], obj["locomotive_pmf"],
                obj["wagon_pmf"])


if __name__ == "__main__":
    unittest.main()
