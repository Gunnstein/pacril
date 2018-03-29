# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import json
import pacril

# Another example where the capabilities of the package are displayed,
# primarily focusing on the object interface, which is more extensive,
# especially for the loads.

# Let us start off by getting the rolling stock for freight trains in the
# Norwegian railway network in the period between 1930  - 60.
rs = pacril.data.NorwegianRollingStock(3, "f")

# What are the locomotives from this period?
for loc in rs.locomotives:
    print(loc.litra, loc.sublitra)

# the wagons are also available through `wagons` attribute. We can draw a
# random train with 25 wagons from the rolling stock.
train = rs.get_train(25)

# and get the load vector of the train
ftrain = train.loadvector

# or of the 6th wagon
fwag6 = train.wagons[5].loadvector

# We can convert the rolling stock, trains, locomotives and wagons
# to json objects with help Pacril JSON Encoder class and the json library
# from the standard library
rs_json = json.dumps(rs, cls=pacril.data.PacrilJSONEncoder)
train_json = json.dumps(train, cls=pacril.data.PacrilJSONEncoder)

# and retrieve them with hlep of the Pacril JSON Decoder
rs2 = json.loads(rs_json, cls=pacril.data.PacrilJSONDecoder)
train2 = json.loads(train_json, cls=pacril.data.PacrilJSONDecoder)

# before we check that the serialization has not changed the objects
np.testing.assert_array_equal(train.loadvector, train2.loadvector)


# The data module contains specific data that has been implemented for certain
# railways. But there are general objects for Loads, Locomotives, TwoAxle,
# Bogie and JacobsWagons.

taw = pacril.TwoAxleWagon(15., 3., 7., 15.*.25)
ptaw, xptaw = taw.p, taw.xp
ftaw = taw.loadvector

ftaw_from_func = pacril.get_loadvector(ptaw, xptaw)

np.testing.assert_array_equal(ftaw, ftaw_from_func)

taw_json = json.dumps(taw, cls=pacril.serialize.PacrilJSONEncoder)
print(taw_json)
