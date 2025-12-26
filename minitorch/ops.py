import numpy as np

#I need this to make the mathematical operations inside tensor.py
class Function:

    def __init__(self, *tensors):

        self.parents = tensors # Take the  Parents

        self.saved_parents = [] # List to save them

    
    
    
    def  save_for_backward(self, *tensors):
        
        self.saved_parents.extend(tensors) # Store the neccesary tensors for Derivative Calculations
    
    
    
    
    @classmethod  #Makes apply use foward and backward of the respective class that have invoked it
    def apply(cls, *tensors):
        from tensor import Tensor # This is intentional to avoid circular importation



        # Save the context object for the backward of the respective class
        ctx = cls(*tensors) #unpack the tensors


        #Take only the numpy array(the tensor "naked")
        input_data = [t.data for t in tensors]


        #foward of the respective class with the np.array
        output_data = ctx.foward(input_data)

        
        #Make it Tensor again
        result = Tensor(output_data)


        #Save the context used to get the result
        result._ctx = ctx 

        return result



    
    #Show you an Error if you didn't wrote the code for the class
    # A Just In Case function
    @staticmethod
    def foward():
        return NotImplementedError 
    
    
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
    


class ReLU(Function):

    @staticmethod
    def foward():
        return None
    
    @staticmethod
    def backward():
        return None
    
