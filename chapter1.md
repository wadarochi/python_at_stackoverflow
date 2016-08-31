# 2016-08-28

* Can a line of Python code know its indentation nesting level? - [52/4]
* Better way to swap elements in list? - [21/11]
* Imported a Python module; why does a reassigning a member in it also affect an import elsewhere? - [17/5]
* How to get a python script to invoke "python -i" when called normally? - [17/5]
* NumPy performance: uint8 vs. float and multiplication vs. division? - [12/3]
* Updating a list within a tuple - [10/1]
* Matching Unicode word boundaries in Python - [10/1]
* How do I release memory used by a pandas dataframe? - [9/2]
* Why does printing a dataframe break python when constructed from numpy empty_like - [9/1]
* Performance between C-contiguous and Fortran-contiguous array operations - [7/2]

## 一行Python代码能知道自己的缩进层级吗？
### [Question](http://stackoverflow.com/questions/39172306/can-a-line-of-python-code-know-its-indentation-nesting-level):
比如：
```python
print(get_indentation_level())

    print(get_indentation_level())

        print(get_indentation_level())
```

我希望得到如下输出：
```python
1
2
3
```

### Answer:
| rank | ▲    | url  |
| :--- | :--: | :--: |
| 1	   | 69   | [url](http://stackoverflow.com/a/39172845/763878)  |



假如你想要的是代码的嵌套层数，而不是行首的空白字符数，问题会变得棘手一些。比如，参考如下代码：
```python
if True:
    print(
get_nesting_level())
```
调用`get_nesting_level`的代码嵌套层数是1，尽管`get_nesting_level`所在的那行代码行首并没有空白字符缩进。

同时，如下代码：
```python
print(1,
      2,
      get_nesting_level())
```
调用`get_nesting_level`的地方，代码嵌套层数为0，但是行首却有空白字符缩进。

在下面的代码中：
```python
if True:
  if True:
    print(get_nesting_level())

if True:
    print(get_nesting_level())
```
两处调用`get_nesting_level`的地方代码嵌套层数不同，但是行首的空白字符缩进却是相同的。

下面的代码中：
```python
if True: print(get_nesting_level())
```
代码的嵌套层数应该算作0还是1呢？从正式的语法记号(Token)`INDENT`和`DEDENT`的角度来看，嵌套层数应该是0，不过你也许不会这么认为。

----
假如你想要获取代码嵌套层数，那么你需要把源文件中，从文件开头到调用`get_nesting_level`处的代码全部标记化(tokenize)，然后从头数`INDENT`和`DEDENT`标记的数量。`tokenize`模块会非常有用：
```python
import inspect
import tokenize

def get_nesting_level():
    caller_frame = inspect.currentframe().f_back
    filename, caller_lineno, _, _, _ = inspect.getframeinfo(caller_frame)
    with open(filename) as f:
        indentation_level = 0
        for token_record in tokenize.generate_tokens(f.readline):
            token_type, _, (token_lineno, _), _, _ = token_record
            if token_lineno > caller_lineno:
                break
            elif token_type == tokenize.INDENT:
                indentation_level += 1
            elif token_type == tokenize.DEDENT:
                indentation_level -= 1
        return indentation_level
```

## 交换列表元素的好办法
### [Question](http://stackoverflow.com/questions/39167057/better-way-to-swap-elements-in-a-list)
我有很多这种类型的列表：
```python
l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```
我想像这样交换列表中的元素位置：
```python
final_l = [2, 1, 4, 3, 6, 5, 8, 7, 10, 9]
```
列表的长度可能不同，但是元素个数都是偶数个。作为一个Python新手，我现在是这么干的：
```python
l =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
final_l = []
for i in range(0, len(l)/2):
    final_l.append(l[2*i+1])
    final_l.append(l[2*i])
```
我知道这样不够Pythonic，希望可以用更高效的写法，比如list comprehension？

### Answer
| rank | ▲    | url  |
| :--- | :--: | :--: |
| 1	   | 54   | [url](http://stackoverflow.com/a/39167545/763878)  |

不需要复杂的逻辑，只需要使用列表切片(slicing)和step：
```python
In [1]: l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

In [2]: l[::2], l[1::2] = l[1::2], l[::2]

In [3]: l
Out[3]: [2, 1, 4, 3, 6, 5, 8, 7, 10, 9]
```

----

*长文慎入(TL;DR -- Too Long; Don't Read)*

我相信大部分读者都熟悉列表切片和多元赋值操作。假如你不熟悉，我在这里尝试解释一下（希望不会把你搞晕）。

如果想了解列表切片，[这里](http://stackoverflow.com/questions/509211/explain-pythons-slice-notation)已经对列表切片概念进行了非常好的解释，比如：
```python
a[start:end] # 返回列表a中从start到end-1的所有元素
a[start:]    # 返回列表a中从start开始的所有元素
a[:end]      # 返回列表a中从开始到end-1的所有元素
a[:]         # 返回整个列表a的拷贝

# 除此之外，列表切片中还可以设置一个步进(step)值，这个步进值可以在上述任一语句中使用

a[start:end:step] # 返回列表a中，从start开始，间隔为step，并且下标(index)不超过end的所有元素
```

让我们再来看看题主(OP)的需求：
```python
 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # list l
  ^  ^  ^  ^  ^  ^  ^  ^  ^  ^
  0  1  2  3  4  5  6  7  8  9    # 列表中元素的下标
l[0]  l[2]  l[4]  l[6]  l[8]      # 第一组 : start=0, step=2
   l[1]  l[3]  l[5]  l[7]  l[9]   # 第二组 : start=1, step=2
-----------------------------------------------------------------------
l[1]  l[3]  l[5]  l[7]  l[9]
   l[0]  l[2]  l[4]  l[6]  l[8]   # 题主想要的结果
```
可以这要获取第一组的元素：`l[::2] = [1, 3, 5, 7, 9]`，第二组可以这样：`l[1::2] = [2, 4, 6, 8, 10]`。

同时我们还需要将两组交换位置，可以使用多元赋值，这么干：
```python
first , second  = second , first
```
具体到这个例子：
```python
l[::2], l[1::2] = l[1::2], l[::2]
```
顺带提一下，如果不想改变l本身，可以先从l拷贝一份新的列表，然后再执行上面的多元赋值：
```python
n = l[:]  # 将n赋值为l的拷贝 (如果不使用 [:], n将仍然指向原来的l)
n[::2], n[1::2] = n[1::2], n[::2]
```
希望这段说明没有把你们任何人弄晕，如果不幸真的把你们弄晕了，请帮忙更新、完善我的这段说明:)