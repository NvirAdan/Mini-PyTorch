#Import the class to use it as a submodule in others projects
from .minitorch.tensor import Tensor as Tensor
from .minitorch import optim as optim
from .minitorch import nn as nn


__all__ = ['Tensor','optim','nn']