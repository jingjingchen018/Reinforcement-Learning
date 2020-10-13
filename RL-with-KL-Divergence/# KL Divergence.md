 # KL Divergence 

参考链接https://hsinjhao.github.io/2019/05/22/KL-DivergenceIntroduction/

## （一）基础概念

（1）分布（distribution）离散概率分布、连续概率分布

（2）事件（event） P(X=1). P(0.95<X<1.05)

本文摘自https://www.jiqizhixin.com/articles/2018-05-29-2

对于一个未知的分布，我们可以用一个已知的分布(e.g. 均匀分布（均匀概率）、二项分布(抛硬币)（n, p）、正态分布())。来表示真实的统计数据。这样只需要知道该分布的参数， 而不用知道真实数据。但是哪种分布可以更好的定量的解释真实分布呢？ 这就是KL divergence 的用武之地。

$P(X=k)=\left(\begin{array}{l}n \\ k\end{array}\right) p^{k}(1-p)^{n-k}$

直观解释： KL 散度是一种衡量两个分布(比如两条线)（近似分布和真实分布）之间的匹配程度的方法。

$D_{K L}(p \| q)=\sum_{i=1}^{N} p\left(x_{i}\right) \log \left(\frac{p\left(x_{i}\right)}{q\left(x_{i}\right)}\right)$

其中 q(x) 是**近似分布**，p(x) 是我们想要用 q(x) 匹配的**真实分布**。直观地说，这衡量的是给定任意分布偏离真实分布的程度。如果两个分布完全匹配，那么![img](https://image.jiqizhixin.com/uploads/editor/3ddd911e-d4d4-4dda-a585-61404f6c911e/1527560009628.png)，否则它的取值应该是在 0 到无穷大（inf）之间。KL 散度越小，真实分布与近似分布之间的匹配就越好。

## (二) KL散度的数学性质

KL散度可以用来衡量两个分布之间的差异。

### 正定性

用Gibbs Iequality证明

### 不对称性

不对称性的直观解释   https://zhuanlan.zhihu.com/p/45131536

$D(p \| q) \neq D(q \| p)$

> 各种散度中，Jensen-Shannon divergence(JS散度)是对称的。



## （三）例子/连续随机变量的KL散度推导

#### （a） 服从一维高斯分布的随机变量的KL散度推导

假设 `p` 和 `q` 均是服从N (μ1,σ21)N (μ1,σ12)和N (μ2,σ22)N (μ2,σ22)的随机变量的概率密度函数 (*probability density function*) ，则从 `q` 到 `p` 的KL散度定义为：



$DKL(p||q)==∫[log(p(x))−log(q(x))]p(x) dx∫[ p(x)log(p(x))−p(x)log(q(x))] dxDKL(p||q)=∫[log⁡(p(x))−log⁡(q(x))]p(x) dx=∫[ p(x)log⁡(p(x))−p(x)log⁡(q(x))] dx$

已知正态分布的概率密度函数(`probability density function`)如下式：

#### (b) 服从多元高斯分布的随机变量KL散度

已在纸上推导过了。







