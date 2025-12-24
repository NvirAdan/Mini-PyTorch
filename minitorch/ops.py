import numpy as np

#I need this to make the mathematical operations inside tensor.py
class Function:

    def __init__(self, *tensors):

        self.parents = tensors # Take the  Parents

        self.saved_parents = [] # List to save them

    
    
    
    def  save_for_backward():
        return None
    
    
    
    
    
    @classmethod  #Makes apply use foward and backward of the respective class that invoke it
    def apply(cls, *args):
        from tensor import Tensor # This is intentional to avoid circular importation



        return None



    @staticmethod
    def foward():
        return NotImplementedError #Show you an Error if you didn't wrote the code for the class
    
    
    @staticmethod
    def backward():
        return NotImplementedError






class Add(Function):

    @staticmethod
    def foward():
        return None
    
    @staticmethod
    def backward():
        return None
    


class Sub(Function):

    @staticmethod
    def foward():
        return None
    
    @staticmethod
    def backward():
        return None
    


class Mul(Function):

    @staticmethod
    def foward():
        return None
    
    @staticmethod
    def backward():
        return None
    


class Matmul(Function):

    @staticmethod
    def foward():
        return None
    
    @staticmethod
    def backward():
        return None
    
