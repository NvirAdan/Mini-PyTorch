import numpy as np 
import torch
from minitorch import Tensor
from minitorch.optim import Adam

def test_python_and_minitorch():

    
    # Numbers suggested by and LLM
    a_data = np.array([[1.0,2.0],[3.0,4.0]])
    b_data = np.array([[0.5,-1.0],[1.0,0.5]])
    c_data = np.array([[0.2,0.2],[0.2,0.2]])



    ###--- MINITORCH ----###

    #Made them a Tensor of Minitorch
    a = Tensor(a_data,requires_grad=True)
    b = Tensor(b_data,requires_grad=True)
    c = Tensor(c_data,requires_grad=True)

    #Foward
    z = (a @ b + c).Softmax()
    loss = z.sum()

    #Backward
    loss.backward()



    ### -----PYTORCH ---### 

    #Made them pytorch tensor
    at = torch.tensor(a_data,requires_grad=True)
    bt = torch.tensor(b_data,requires_grad=True)
    ct = torch.tensor(c_data,requires_grad=True)

    #Foward
    zt = torch.nn.functional.softmax(at @ bt + ct, dim=-1)
    losst= zt.sum()

    #Backward
    losst.backward()

    #---RESULTS---##

    print("Results:")
    print(f"Loss minitorch: {loss.data}")
    print(f"Loss Pytorch: {losst.item()}")

    #Gradients "a"
    np.testing.assert_allclose(a.grad, at.grad.numpy(), atol = 1e-6)
    print("Gradient A is the same")

    #Gradients "b"
    np.testing.assert_allclose(b.grad, bt.grad.numpy(), atol= 1e-6)
    print("Gradient B is the same")



if __name__ == "__main__":
    try:
        test_python_and_minitorch()
        print("YAY finally it works,now I got brain damage")
    except Exception as e:
        print(f"Keep working mf:{e}")