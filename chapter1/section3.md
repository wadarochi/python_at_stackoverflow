## import一个Python模块，为什么对其属性的重新赋值会影响到其它同样import了这个模块的地方？
### [Question](http://stackoverflow.com/questions/39144498/imported-a-python-module-why-does-a-reassigning-a-member-in-it-also-affect-an-i):

我看到Python有一种无法理解的行为，考虑如下布局：
```
project
|   main.py
|   test1.py
|   test2.py
|   config.py
```

main.py:
```python
import config as conf
import test1
import test2

print(conf.test_var)
test1.test1()
print(conf.test_var)
test2.test2()
```

test1.py:
```python
import config as conf

def test1():
    conf.test_var = 'test1'
```

test2.py:
```python
import config as conf

def test2():
    print(conf.test_var)
```

config.py:
```python
test_var = 'initial_value'
```

运行`python main.py`获得如下输出：
```
initial_value
test1
test1
```

### Answer:
| rank | ▲    | url  |
| :--- | :--: | :--: |
| 1	   | 25   | [url](http://stackoverflow.com/a/39144686/763878)  |

Python会缓存import进来的模块。第二次`import`不会reload这个模块。
