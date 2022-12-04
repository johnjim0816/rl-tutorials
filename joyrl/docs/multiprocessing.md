[multiprocessing官方教程-Python](https://docs.python.org/zh-cn/3/library/multiprocessing.html)

容易陷入的误区：

* 电脑的CPU核数不等于支持的进程数，实际上能够支持的进程数更多，一般每个核支持两个进程
* 进程与线程也有区别

执行下列代码可查看电脑能够支持的最大进程数：
```python
import multiprocessing as mp
print(mp.cpu_count())
```