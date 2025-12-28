import numpy as np
from operations import Add,Sub,Mul,Matmul,Sum

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

    
    
    def sum(self):
        #This doesn't need to prove that is a tensor like the other ones.

        return Sum.apply(self)
    
    
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
                        walk(parent)

                # Once we have already visited all its parents we add it to the list
                topo.append(t)

        #We start al over again from the current tensor (normally the loss/gradient).
        walk(self)

        return topo

    
    
    
    def backward():
        return None
    

        
        
            

