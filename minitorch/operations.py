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
        from .tensor import Tensor # This is intentional to avoid circular importation



        # Save the context object for the backward of the respective class
        ctx = cls(*tensors) #unpack the tensors

        
        #Take only the numpy array(the tensor "naked")
        input_data = [t.data if isinstance(t, Tensor)else t for t in tensors]#To avoid issues with .data fo the arguments


        #foward of the respective class with the np.array
        output_data = ctx.foward(input_data)

        
        #Make it Tensor again
        result = Tensor(output_data)


        #Save the context used to get the result
        result._ctx = ctx 

        #Assing gradient if one the parents requires gradient
        result.requires_grad = any(t.requires_grad for t in tensors)

        return result



    
    #Raise an Error if you didn't wrote the code for the class
    # A "Just In Case" functionality
    
    def foward(self, *args):
        raise NotImplementedError("Olvidaste el Foward")
    
    
    
    def backward(self, *args):
        raise NotImplementedError("Olvidaste el Backward")




#Complicated to simplified 


# This is key to avoid broadcasting errors,these broadcasting are different to the ones
# that numpy manages,these errors appears before reaching numpy.
def unbroadcast(grad,original_shape):
    
    # First colapse, we sum the "extra dimensions" that comes in the gradient in order
    # to be able to have the same dimensions(original shape) of the parent_tensor
    while len(grad.shape) > len(original_shape):
        # the while is just in the case that the gradient is a tensor4D (for example) and the parent is
        # a tensor2D it will keep colapsing until reach the same dimensions as the parent.
        
        
        # the summatory of the dimensions is the new value of the dimension,always respecting the position.
        grad = grad.sum(axis=0)


    
    # Second colapse,the first colapse make them have the same dimensions,this one make them have the same
    # value in every dimension for example(parent_dim(1,512) and gradient(10,512))this will colapse the 10 of the gradient
    # into 1 to make them have the same value. 
    for i,dim in enumerate(original_shape):
        
        
        # if the original value in the parent is 1 we can asumme that numpy streched the value in the foward pass
        if dim == 1:
            
            
            # imagine a excel sheet,here in every column we sum all the values of the rows in that column to make one row
            # with that sum of values
            grad = grad.sum(axis=i, keepdims=True)

    
    # now just return the tensor
    return grad







class Add(Function):

   
    def foward(self,input_data):

        #Take the values
        x, y = input_data


        #we save in the list the tensor used (shape for unbroadcasting)
        self.save_for_backward(x.shape,y.shape)
        

        return x + y
    
    
    def backward(self, grad_output):

        #Alwarys check th gradient because if it is None it will spread like a virus through backprop
        if grad_output is None:
            return None, None
        
        # Get the original_shapes
        x_shape, y_shape = self.saved_parents


        # Addition "spread" the gradient equally
        return unbroadcast(grad_output,x_shape),unbroadcast(grad_output,y_shape)
    


class Sub(Function):

    
    def foward(self,input_data):

        #Extract the values
        x, y = input_data

        #Save for backward (shape for unbrocasting)
        self.save_for_backward(x.shape,y.shape)

        return x - y
    
    
    def backward(self, grad_output):

        #Check grad
        if grad_output is None:
            return None,None
        

        # Get the original_shapes
        x_shape , y_shape = self.saved_parents


        # x - y    =>   dx = 1   ;   dy = -1 
        return unbroadcast(grad_output,x_shape), unbroadcast(-grad_output,y_shape) #unbroadcasting
    


class Mul(Function):

    
    def foward(self, input_data):

        # Extract data
        x, y = input_data

        #Save for backward (we need the data to not onlythe shape like Add or Sub)
        self.save_for_backward(x,y)

        
        return x * y
    
    
    def backward(self,grad_output):

        #Check gradient
        if grad_output is None:
            return None,None
        

        # Extract the parents
        x, y = self.saved_parents

        
        # The derivative of X is Y and the derivative of Y is X
        x_grad = grad_output * y
        y_grad = grad_output * x
        
        
        #returning Unbroadcasted tensor(we only pass the shape to the unbroadcast function)
        return unbroadcast(x_grad,x.shape), unbroadcast(y_grad,y.shape) 
    


class Matmul(Function):

    
    def foward(self,input_data):

        #Extract data
        x, y = input_data

        #save for backward (don´t need unbroadcasting)
        self.save_for_backward(x,y)


        return x @ y 
    
    
    def backward(self, grad_output):

        #Check gradient
        if grad_output is None:
            return None, None



        # Get parents
        x, y = self.saved_parents

        #Don´t forget the Transpose in order to actually be able to do the operation.
        #Like in the backward  of the Mul class we need the data inside the Tensor.
        grad_x = grad_output @ y.T

        grad_y = x.T @ grad_output

        return grad_x,grad_y
    


class ReLU(Function):

    
    def foward(self,input_data):
        
        # we just need one tensor
        x = input_data[0]

        self.save_for_backward(x)

        return np.maximum(0,x)
    
    
    def backward(self, grad_output):

        # Check Gradient but the operation that only have 1 parent gives back just 1 None
        if grad_output is None:
            return None
        



        # Just one because ReLU only have one parent
        x = self.saved_parents[0]

        # Is like a door where only the number that was greater than zero
        # let the gradient pass through there.
        # Here we are using a mask of booleans (x > 0) to do the trick.
        grad_x = grad_output * (x > 0) 

        return grad_x
    

class Sum(Function):

    def foward(self, input_data):

        x = input_data[0]
        
        # We save only the original form
        self.save_for_backward(x)
        
        #Make a list of the summatory and then a numpy array of it
        return np.array([np.sum(x)])
    
    def backward(self, grad_output):

        #Check Gradient
        if grad_output is None:
            return None
        



        
        # Get back the original form of the tensor before the summatory.
        x = self.saved_parents[0]

        # Make a grid of 1s with the form of the tuple  and multiply it by the gradient.
        return np.ones(x.shape) * grad_output[0]



class Reshape(Function):

    def foward(self, input_data):

        # This is the current form of the data
        x_data = input_data[0]

        
        #This is the new shape that we want
        new_shape = input_data[1]

        
        #We save the original form
        self.save_for_backward(x_data.shape)

        
        # And retrieve the new form of the data
        return x_data.reshape(new_shape)


    def backward(self, grad_output): 

        #Check Gradient
        if grad_output is None:
            return None,None
        



        #We take back the original for of the data from the parents
        original_shape = self.saved_parents[0]


        # And the "derivative" of the gradient is just the original form
        return grad_output.reshape(original_shape), None # Because we use tuples for the tensor.backward()



class Transpose(Function):

    def foward(self, input_data):

        #We take the current form
        x_data = input_data[0]

        #We save it(not actually needed but for a just in case scenario)
        self.save_for_backward(x_data.shape)


        # And return the transpose 
        return x_data.T
    


    
    def backward(self, grad_output):

        #Check Gradient
        if grad_output is None:
            return None
        



        #And the "derivative" its just another Transpose
        return grad_output.T
    
    


class Softmax(Function):

    def foward(self, input_data):

        # First we take the data
        x_data = input_data[0]

        #If the numbers are too high,exp explodes
        # for that reason we take the highest number inside our data
        x_max = np.max(x_data, axis= -1, keepdims=True)

        #And substract it to the rest in order to do the exponentiation
        e_numbers = np.exp(x_data - x_max)


        #Now the result is only the exponents divided by the summatory of the exponents
        out = e_numbers / np.sum(e_numbers, axis= -1, keepdims=True)


        #as always Save the result
        self.save_for_backward(out)

        # And return it
        return out
    

    def backward(self, grad_output):

        #Check Gradient
        if grad_output is None:
            return None
        






        #Take back the parents
        out = self.saved_parents[0]


        # The derivative is just the summatory of the gradient multiplied by the out of softmax in the foward pass
        summatory_grad_and_out = np.sum(grad_output * out ,axis = -1, keepdims=True)

        #And then to that we do the out of softmax multiplied by the gradient subtracted by the previous summatory
        x_grad = out * (grad_output - summatory_grad_and_out)

        # and return it
        return x_grad
    



class Pow(Function):

    def foward(self, input_data):

        # Our current value
        x = input_data[0]

        #The value that we want to apply
        n = input_data[1]


        #Save them

        self.save_for_backward(x, n)

        # return the potentiation
        return x ** n
    
    def backward(self, grad_output):


        #Check Gradient
        if grad_output is None:
            return None,None
        





        #Take the parents back
        x, n = self.saved_parents

        # The derivative of x^n is n * x^(n-1) for the conmmutation rule

        return grad_output * (n * (x ** (n-1))), None # Because we use tuples
    
    

class Log(Function):

    def foward(self,input_data):

        #We take the value
        x = input_data[0]

        #Then we save it
        self.save_for_backward(x)

        #This is only to avoid log(0) which is -infinite
        eps = 1e-15

        return np.log(x + eps)
    

    def backward(self, grad_output):

        #Check the gradient
        if grad_output is None:
            return None
        
        #Extract the value
        x = self.saved_parents[0]

        #Again, to avoid issues
        eps = 1e-15
        
        
        #The derivate of log(x) is 1/x
        return grad_output/(x + eps)
