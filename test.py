import sys
sys.path.append('python/needle')
import needle as ndl
from needle import NDArray
import numpy as np


a = np.arange(6**2).reshape((6, 6))

A = NDArray(a)
B = A[:, 0:6:2]
print(A)
print(type(A))
print(A.device)
print(B)

C = B[1:5:3, :]
print(C)
print(C._handle == A._handle)

D = B[1:5:3, 0:3:2]
print(D)
print(D._handle == A._handle)
print(D.strides)
print(D._offset)

E = D.permute((1, 0))
print(E)
print(E._handle == A._handle)
print(E.strides)

F = E.reshape((4, 1)) #seem must create a new handle
print(F)
print(F._handle == F._handle)
print(F.shape)

G = F.broadcast_to((4, 4))
print(G)
print(G._handle == G._handle)