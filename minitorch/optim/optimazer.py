#as always numpy carrying
import numpy as np

# Old trusty,just in case and because is the only one that I have used it
class SGD:
    def __init__(self, params, lr=0.01):
        
        self.params = params
        self.lr = lr

    def step(self):

        # for every parameter
        for p in self.params:
            
            # if it have gradient
            if p.grad is not None:

                # Subtract the gradient multipliying by the learning rate to the parameter
                p.data -= self.lr * p.grad

    def zero_grad(self):

        # Now you can get rid of the gradient because you've already used it
        for p in self.params:
            p.grad = None


# War machine right here,This is the one that I'll use for the Tiny Transformer
class Adam:
    #Take note here are the hyper-parameters beta1 and beta2 and eps is just for the computer
    #so it doesn't assume values that are too low as 0
    def __init__(self,params,lr=0.001,beta1=0.9,beta2=0.999,eps=1e-8):

        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        #clock
        self.t = 0

        #We initialize momentum for every parameter

        #Momentum
        self.m = [np.zeros_like(p.data) for p in self.params]
        #Variability
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        # for every step +1 to the clock
        self.t += 1

        #looping with enumerate (index and value)
        for i, p in enumerate(self.params):
            
            #Check if it have grad
            if p.grad is not None:

                #This are The Exponential Moving Average(EMA,I search it out I promise its called like that)
                # They are like measure the tempeture in a city,you need to know the temperature of previous days
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad**2)

                #Bias correction because in the first time,the clock previously have a 0 and that will influence
                #in the result of the actualization of the params if we don't do this
                m_hat = self.m[i] / ( 1 - self.beta1**self.t)
                v_hat = self.v[i] / ( 1 - self.beta2**self.t)

                #And here is the actual adaptative actualization
                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        # as always clean after you work
        for p in self.params:

            p.grad = None


