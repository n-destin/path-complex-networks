import inspect
from typing import Callable, List, Optional
from torch_geometric.inspector import Inspector
from collections import OrderedDict



'''Testing complex inspector class'''

def function(name: str, age: int) -> str:
    return '{} is {} old'.format(name, age)

class ComplexInspector(Inspector):
    def __init__(self, cls):
        super().__init__(cls)

    def implements(self, function, cls):
        if cls.__name__ == "MessagePassing":
            return False 
        if function.__name__ in cls.__dict__.keys():
            return True 
        return any(self.implements(_cls, function) for _cls in cls.__bases__)
    

    def inspect(self, function : Callable, remove_top_n : int = 0):
        params = inspect.signature(function).parameters()
        params = OrderedDict(params)

        for _ in range(remove_top_n):
            params.popitem(last=False)
        self.params[function.__name__] = params