import numpy as np
from operations import Add,Sub,Mul,Matmul

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
        #Check if is a Tensor,if not make it.
        if not isinstance(other, Tensor):
            other = Tensor(other)
            

        return Add.apply(self, other) # operations.py tools
    
    
    
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        return Sub.apply(self, other)
    
    
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        return Mul.apply(self, other)
    
    
    
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        return Matmul.apply(self, other)

    
    
    
    def sum():
        return None
    
    def backward():
        return None
    

        
        
            

