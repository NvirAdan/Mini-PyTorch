import numpy as np


# This is the core of the minitorch,the central to manage the data.
class Tensor:

    # Creation of the tensor with our own characteristics
    def __init__(self,data,requires_grad=False):

        #For us to be sure that data is always a numpy array
        if isinstance(data,(list,tuple)):
            data = np.array(data)

        self.data = data
        self.requires_grad = requires_grad

        self.grad = None
        
        self._ctx = None

    
    
    
    # Mathematical operations for the Tensor that we have created
    def __add__(self,other):

        #Local Importations to avoid issues
        from .operations import Add

        #Check if is a Tensor,if not make it.
        if not isinstance(other, Tensor):
            other = Tensor(other)
            

        return Add.apply(self, other) # operations.py tools
    
    
    
    def __sub__(self, other):
        
        from .operations import Sub


        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        return Sub.apply(self, other)
    
    
    
    def __mul__(self, other):

        from .operations import Mul


        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        return Mul.apply(self, other)
    
    
    
    def __matmul__(self, other):

        from .operations import Matmul


        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        return Matmul.apply(self, other)

    
    
    def __pow__(self, n):

        from .operations import Pow


        #Doesn't need to check if is a tensor it can handle it
        return Pow.apply(self, n)
    
    
    
    
    def sum(self):

        from .operations import Sum


        #This doesn't need to prove that is a tensor like the other ones.

        return Sum.apply(self)
    
    
    
    def reshape(self,new_shape):

        from .operations import Reshape

        return Reshape.apply(self,new_shape)
    


    def Transpose(self):

        from .operations import Transpose


        return Transpose.apply(self)
    
    @property # to avoid use constantly "()" and write the whole word "Transpose"
    def T(self):

        return self.Transpose()
    
    
    
    
    def Softmax(self):

        from .operations import Softmax

        return Softmax.apply(self)
    
    
    
    
    
    
    





    
    
    #Here is where we do our topological sort of the DAG (Directed Acyclic Graph).
    #Is like a creation of roadmap for our backward function
    def _build_topo(self):
        
        #Here we save the tensor in order
        topo = [] 

        #This is to not repeat the same tensor        
        visited = set()

        #Intern function for DFS (Deep First Search).
        def walk(t):

            #If is already visited we do nothing.
            if t not in visited:
                visited.add(t)

                # If the tensor have context if because is a result of an operation.
                if t._ctx is not None:
                    # Recursion, we visit all its parents before add it to the list.
                    for parent in t._ctx.parents:
                        if isinstance(parent, Tensor):#Validation "just in case"
                            walk(parent)

                # Once we have already visited all its parents we add it to the list
                topo.append(t)

        #We start al over again from the current tensor (normally the loss/gradient).
        walk(self)

        return topo

    
    
    
    def backward(self):

        #The Gradient of the output es always 1
        if self.grad is None:
            self.grad = np.ones_like(self.data)#Same size as the final tensor

        
        #We call the topological sort function for our tensor
        topo = self._build_topo()

        #We gonna follow the topo but in reverse (because is a backpropagation)
        for t in reversed(topo):
            
            #If the tensor was created by a class(por example:Add,Sub)
            if t._ctx is not None:
                
                #We call its respective backward of the class
                grads = t._ctx.backward(t.grad)

                #Making sure it is a Tuple because in the backprop of the tensor I only use tuples
                if not isinstance(grads, tuple):
                    grads = (grads,)
                
                # We zip it the current "node/tensor" in the topo and its respective grads
                # in order for every operations have the parent and the gradient.
                for parent,g in zip(t._ctx.parents, grads):
                    #Self explanatory but is specifically needed for it to know when to stop
                    if isinstance(parent, Tensor) and parent.requires_grad and g is not None:
                        
                        if parent.grad is None:
                        
                            #If it doesn't have gradient first we create the "container"
                            parent.grad = np.zeros_like(parent.data)

                        #Know the fill the container with the gradient (+= is important to not overlap)
                        parent.grad += g
                

            








        
    

        
        
            

