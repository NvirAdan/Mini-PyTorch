import numpy as np
from ops import Add,Sub,Mul,Matmul

# This is the core of the minitorch,the central to manage the data.
class Tensor:

    # Creation of the tensor with our own characteristics
    def __init__(self,data,requires_grad=False):

        self.data = data
        self.requires_grad = requires_grad

        self.grad = None
        
        self._ctx = None

    # Mathematical operations for the Tensor that we have created
    def __add__(self,other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
            

        return None # ops.py tools
    
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        return None
    
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        return None
    
    def __mul__(self, other):
        if not isinstance(self, other):
            other = Tensor(other)
        
        return None

    def sum():
        return None
    
    def backward():
        return None
    

        
        
            

