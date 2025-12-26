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

        #Assing gradient if one the parents requires gradient
        result._requires_grad = any(t.requires_grad for t in tensors)

        return result



    
    #Raise an Error if you didn't wrote the code for the class
    # A "Just In Case" functionality
    
    def foward(self, *args):
        raise NotImplementedError("Olvidaste el Foward")
    
    
    
    def backward(self, *args):
        raise NotImplementedError("Olvidaste el Backward")






class Add(Function):

   
    def foward(self,input_data):

        #Take the values
        x, y = input_data


        #we save in the list the tensor used
        self.save_for_backward(x,y)
        

        return x + y
    
    
    def backward(self, grad_output):

        # Addition "spread" the gradient equally
        return grad_output,grad_output
    


class Sub(Function):

    
    def foward(self,input_data):

        x, y = input_data

        self.save_for_backward(x,y)

        return x - y
    
    
    def backward(self, grad_output):

        # x - y    =>   dx = 1   ;   dy = -1 
        return grad_output, -grad_output
    


class Mul(Function):

    
    def foward(self, input_data):

        x, y = input_data

        self.save_for_backward(x,y)

        return x * y
    
    
    def backward(self,grad_output):

        x, y = self.saved_parents

        # The derivative of X is Y and the derivative of Y is X
        return grad_output * y.data,grad_output * x.data
    


class Matmul(Function):

    
    def foward(self,input_data):

        x, y = input_data

        self.save_for_backward(x,y)

        return x @ y 
    
    
    def backward(self, grad_output):

        x, y = self.saved_parents

        #Don´t forget the Transpose in order to actually be able to do the operation.
        #Like in the backward  of the Mul class we need the data inside the Tensor.
        grad_x = grad_output @ y.data.T

        grad_y = x.data.T @ grad_output

        return grad_x,grad_y
    


class ReLU(Function):

    
    def foward(self,input_data):
        
        # we just need one tensor
        x = input_data[0]

        self.save_for_backward(x)

        return np.maximum(0,x)
    
    
    def backward(self, grad_output):

        # Just one because ReLU only have one parent
        x = self.saved_parents[0]

        # Is like a door where only the number that was greater than zero
        # let the gradient pass through there.
        # Here we are using a mask of booleans (x > 0) to do the trick.
        grad_x = grad_output * (x > 0) 

        return grad_x
    