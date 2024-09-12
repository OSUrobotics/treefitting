#!/usr/bin/env python3
import json


class BSplineFitParams(dict):
    _param_names = {"end derivs": False,      # if True, set the end derivatives of the spline
                    "weight ctrl pts": 0.0,   # if non-zero, add weighted control points to fit}
                    "outlier ratio": 0.1,     # percentage of points that are allowed outside of a "good" fit
                    "inlier threshold": 0.1,  # Numerical value; distance at which the point is considered an inlier
                    "average fit": 0.1}       # Numerical value; average error allowed across all points

    def __init__(self, json_string="{}"):
        super().__init__(self)
        for k, v in BSplineFitParams._param_names.items():
            self[k] = v
        self.load_json(json_string)

    def is_param(self, str_name):
        return str_name in self.keys()

    def get_param(self, str_name):
        """ check if a parameter exists in the dictionary and return it if it does, None on failure
        :param str_name: parameter name
        :return: parameter value or None
        """
        return self.get(str_name)

    def add_param(self, str_name, val):
        self[str_name] = val

    def update_param(self, str_name, val):
        self[str_name] = val

    def json_dump(self):
        return json.dumps(self, indent=4)

    def load_json(self, json_str: str):
        dict_load = json.loads(json_str)
        for k, v in dict_load:
            self[k] = v

    def __repr__(self):
        return f'{self.json_dump()}'

    def __deepcopy__(self, memo):
        # Create a new instance of BSplineFitParams
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
