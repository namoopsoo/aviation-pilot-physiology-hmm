import numpy as np
import json

import unittest
import mytf.validation as mv

class TestFoo(object):
    def test_basic(self):


        json.dumps({'blah': np.float32(3)},
                cls=mv.JSONCustomEncoder)

