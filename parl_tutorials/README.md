需要额外安装：
```bash
pip install parl==2.0.5

pip install paddlepaddle-gpu==2.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install paddlepaddle==2.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果与notebook相关模块发生版本冲突，则建议新建一个Conda环境，执行：
```bash
pip install -r parl_requirements.txt
```
然后再按照上文的提示安装```parl```和```paddle```