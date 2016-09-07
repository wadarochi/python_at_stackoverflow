## Numpy性能: uint8 vs. float，乘法运算 vs. 除法运算？
### [Question](http://stackoverflow.com/questions/39104562/numpy-performance-uint8-vs-float-and-multiplication-vs-division):

我发现我的一个脚本的运行时间减半仅仅是因为我把其中的乘法运算改成了除法运算。

为了搞清楚这个问题，我写了一个小例子：

```python
import numpy as np                                                                                                                                                                                
import timeit

# uint8 array
arr1 = np.random.randint(0, high=256, size=(100, 100), dtype=np.uint8)

# float32 array
arr2 = np.random.rand(100, 100).astype(np.float32)
arr2 *= 255.0


def arrmult(a):
    """ 
    mult, read-write iterator
    """
    b = a.copy()
    for item in np.nditer(b, op_flags=["readwrite"]):
        item[...] = (item + 5) * 0.5

def arrmult2(a):
    """ 
    mult, index iterator
    """
    b = a.copy()
    for i, j in np.ndindex(b.shape):
        b[i, j] = (b[i, j] + 5) * 0.5

def arrmult3(a):
    """
    mult, vectorized
    """
    b = a.copy()
    b = (b + 5) * 0.5

def arrdiv(a):
    """ 
    div, read-write iterator 
    """
    b = a.copy()
    for item in np.nditer(b, op_flags=["readwrite"]):
        item[...] = (item + 5) / 2

def arrdiv2(a):
    """ 
    div, index iterator
    """
    b = a.copy()
    for i, j in np.ndindex(b.shape):
           b[i, j] = (b[i, j] + 5)  / 2                                                                                 

def arrdiv3(a):                                                                                                     
    """                                                                                                             
    div, vectorized                                                                                                 
    """                                                                                                             
    b = a.copy()                                                                                                    
    b = (b + 5) / 2                                                                                               




def print_time(name, t):                                                                                            
    print("{: <10}: {: >6.4f}s".format(name, t))                                                                    

timeit_iterations = 100                                                                                             

print("uint8 arrays")                                                                                               
print_time("arrmult", timeit.timeit("arrmult(arr1)", "from __main__ import arrmult, arr1", number=timeit_iterations))
print_time("arrmult2", timeit.timeit("arrmult2(arr1)", "from __main__ import arrmult2, arr1", number=timeit_iterations))
print_time("arrmult3", timeit.timeit("arrmult3(arr1)", "from __main__ import arrmult3, arr1", number=timeit_iterations))
print_time("arrdiv", timeit.timeit("arrdiv(arr1)", "from __main__ import arrdiv, arr1", number=timeit_iterations))  
print_time("arrdiv2", timeit.timeit("arrdiv2(arr1)", "from __main__ import arrdiv2, arr1", number=timeit_iterations))
print_time("arrdiv3", timeit.timeit("arrdiv3(arr1)", "from __main__ import arrdiv3, arr1", number=timeit_iterations))

print("\nfloat32 arrays")                                                                                           
print_time("arrmult", timeit.timeit("arrmult(arr2)", "from __main__ import arrmult, arr2", number=timeit_iterations))
print_time("arrmult2", timeit.timeit("arrmult2(arr2)", "from __main__ import arrmult2, arr2", number=timeit_iterations))
print_time("arrmult3", timeit.timeit("arrmult3(arr2)", "from __main__ import arrmult3, arr2", number=timeit_iterations))
print_time("arrdiv", timeit.timeit("arrdiv(arr2)", "from __main__ import arrdiv, arr2", number=timeit_iterations))  
print_time("arrdiv2", timeit.timeit("arrdiv2(arr2)", "from __main__ import arrdiv2, arr2", number=timeit_iterations))
print_time("arrdiv3", timeit.timeit("arrdiv3(arr2)", "from __main__ import arrdiv3, arr2", number=timeit_iterations))
```

运行这个脚本，得到如下运行时间测量结果：
```txt
uint8 arrays
arrmult   : 2.2004s
arrmult2  : 3.0589s
arrmult3  : 0.0014s
arrdiv    : 1.1540s
arrdiv2   : 2.0780s
arrdiv3   : 0.0027s

float32 arrays
arrmult   : 1.2708s
arrmult2  : 2.4120s
arrmult3  : 0.0009s
arrdiv    : 1.5771s
arrdiv2   : 2.3843s
arrdiv3   : 0.0009s
```

我一直认为乘法运算的复杂度要低于除法运算，但是对于`uint8`，除法运算的性能是乘法运算的两倍。这会不会是因为`* 0.5`是需要计算浮点数的乘法，之后再将结果转换为一个整型？

至少再上面的浮点运算测量结果中，乘法运算看起来要比除法运算更快，这是正确的吧？

为什么`uint8`的乘法运算要比`float32`的更耗？我觉得8比特的无符号整型在运算时应该远远快于32比特的浮点数呀。

有没有哪位能阐明其中的道理？

*EDIT*: bala bala

*EDIT2* bala bala

### Answer:
| rank | ▲    | url  |
| :--- | :--: | :--: |
| 1	   | 5    | [url](http://stackoverflow.com/a/39174057/763878) |


