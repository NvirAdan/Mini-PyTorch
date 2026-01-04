#Import the class to use it as a submodule in others projects
from .tensor import Tensor as Tensor
from . import optim as optim
from . import nn as nn


__all__ = ['Tensor','optim','nn']