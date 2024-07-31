#!/usr/bin/env python3
import json
import numpy as np

class fit_params:
    def __init__(
        self,
        json_string = None,
    ):
        self.load_json(json_string)

    def is_param(self, str):
        return str in self.__dict__.keys()

    def get_param(self, str):
        return self.__dict__[str]
    
    def add_param(self, str, val):
        self.__dict__[str] = val

    def update_param(self, str, val):
        self.__dict__[str] = val

    def json_dump(self):
        return json.dumps(self.__dict__, indent=4)

    def load_json(self, json_str: str):
        self.__dict__ = json.loads(json_str)
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key], list):
                self.__dict__[key] = np.array(self.__dict__[key])

if __name__ == "__main__":
    fp = fit_params(
        1.0, residuals=np.array([0.6, 0.3, 0.2]), ts=np.array([0.2, 0.7, 1.0])
    )
    print(fp.json_dump())
    pass
