# \brucedell\py4\np.py
# from NumPy section of course
import numpy as np
list1=[20, 35, 77, 42, 3, 51]
arr=np.array(list1)     # 1-d array or vector
print(arr)
print(arr.min())        # 3
print(arr.max())        # 77
print(arr.sum())        # 228
print(arr.mean())       # 38
print(arr.argmin())     # 4
print(arr.argmax())     # 2arr
print(arr.size)         # 6
print(arr.dtype)        # int32
print(arr.shape)        # 1x6
print(type(arr))        # ndarray

bool_arr=arr > 20
print(bool_arr)
print(arr[bool_arr])    # conditional selection
print(arr[arr > 20])    # preferred

# A 2-d array is known as a matrix
arr2d=np.array([[10,20,30],[40,50,60],[70,80,90]])
print(type(arr2d))
print(arr2d)
print(arr2d[0][2])      # double bracket notation
print(arr2d[0,2])       # single bracket notation, preferred
print(arr2d[:2,1:])     # rows 0,1 & cols 1,2
print(arr2d[0])         # just row 0
print(arr2d[0].shape)   # 1x3
print(arr2d[:,0])       # just col 0
print(arr2d[:,0].shape) # 3x1

ar=np.arange(6)
print(ar+ar, ar-ar)     # array to array operations
print(ar*ar, ar/ar)     # note warning for 0/0
print(ar*100)           # scalar to array operations
print(np.exp(ar))       # see list of ufuncs in docs
print(np.log(ar))       # note first value -inf
print(1/0)              # note that this is an error
