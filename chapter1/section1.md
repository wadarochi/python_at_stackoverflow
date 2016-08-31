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
