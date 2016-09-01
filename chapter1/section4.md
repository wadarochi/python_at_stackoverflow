## 如何让python脚本自行调用"python -i"
### [Question](http://stackoverflow.com/questions/39155928/how-to-get-a-python-script-to-invoke-python-i-when-called-normally):

我喜欢使用`python -i script.py`这样的方式来运行我的一份python脚本，这可以在运行脚本之后进入交互模式，于是我可以在python交互模式中操作`script.py`的运行结果。

有没有可能直接在脚本中调用到这个选项(-i)，这样我就可以在仅仅使用`python script.py`的情况下，也能在script.py运行完之后进入交互模式？

当然，我也可以加上`-i`选项，或者上面这个需求很难实现的话，我也可以写一个shell脚本来调用`-i`选项。

### Answer:
| rank | ▲    | url  |
| :--- | :--: | :--: |
| 1	   | 19   | [url](http://stackoverflow.com/a/39155981/763878)  |

在`script.py`中，设置环境变量[PYTHONINSPECT](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONINSPECT)为任意非空字符串。Python会在脚本运行程序结束的时候检测这个环境变量并进入交互模式。
```python
import os
# 这行代码可以至于脚本文件的开头或者结尾处, 不像code.interact必须要挡在结尾处
os.environ['PYTHONINSPECT'] = 'TRUE'
```

### 其它解决方案
#### 将`-i`选项写到shebang里面去：
```python
#!/usr/bin/python -i
this = "A really boring program"
```
但是像上面这样，就需要对这个脚本加可执行权限，并且不能再`python script.py`，必须要`./script.py`。

#### 使用IPython
```python
import IPython
IPython.embed()
```

但是与上文所述的code.interact有相同的问题，需要放在脚本的末尾，不过IPython的shell要赞的多。
