# MERL 

Maximum entropy RL 

知乎上有很多关于soft AC 的论文。本篇文章摘自知乎

https://zhuanlan.zhihu.com/p/146460198

## 1.整体理解

Soft Actor Critic出自BAIRhttps://bair.berkeley.edu 伯克利人工智能实验室和Google Brain，作者是Tuomas Haarnoja，他是[Pieter Abbeel](https://link.zhihu.com/?target=https%3A//people.eecs.berkeley.edu/~pabbeel/)和[Sergey Levine](https://link.zhihu.com/?target=https%3A//people.eecs.berkeley.edu/~svlevine/)的学生。他们对于该算法发表了两篇paper（p1和p2）。他们说SAC是第一个**off-policy + actor critic + maximum entropy的RL算法**。与以往算法相比，该算法主要解决的问题是stable和sample inefficiency，其中stable是相对于Policy Gradient类算法，如DQN，DDPG等，inefficiency是相对于on-policy算法，如TRPO，PPO等算法。

p1.[Soft Actor-Critic:Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1801.01290)

```
@article{haarnoja2018soft,
  title={Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor},
  author={Haarnoja, Tuomas and Zhou, Aurick and Abbeel, Pieter and Levine, Sergey},
  journal={arXiv preprint arXiv:1801.01290},
  year={2018}
}
```



p2.[Soft Actor-Critic Algorithms and Applications](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1812.05905)

```
@article{haarnoja2018soft,
  title={Soft actor-critic algorithms and applications},
  author={Haarnoja, Tuomas and Zhou, Aurick and Hartikainen, Kristian and Tucker, George and Ha, Sehoon and Tan, Jie and Kumar, Vikash and Zhu, Henry and Gupta, Abhishek and Abbeel, Pieter and others},
  journal={arXiv preprint arXiv:1812.05905},
  year={2018}
}
```

该作者还有一篇

```
@article{haarnoja2017reinforcement,
  title={Reinforcement learning with deep energy-based policies},
  author={Haarnoja, Tuomas and Tang, Haoran and Abbeel, Pieter and Levine, Sergey},
  journal={arXiv preprint arXiv:1702.08165},
  year={2017}
}
```

文章对SAC的方法总结了有三个要点：

- Actor-critic框架（由两个网络分别近似policy和value function/ q-function）
- Off-policy（提高样本使用效率）
- Entropy来保证stable和exploration

注：要理解SAC算法关键是要理解其中的soft 和exploration.

## 2 基础说明

### 2.1 Actor-critic框架

如何理解actor-critic呢？这里有policy iteration的思想，简单描述就是从一个基础policy开始，先对其每个action计算value，然后根据max value更新policy，目的是以value为指向使得policy逐步提升，可以概括为two-step的闭环。

### 2.2 为何off-policy可提高样本效率

先理解on-policy和off-policy的区别，**on-policy是指样本的生成是基于当前策略**，而**off-policy是样本生成与策略无关**。

~在定义上，与环境交互的policy ~

### 2.3 Entropy项的作用

一般RL的目标是学习一个policy，使得累积reword期望值最大，即：

![img](https://pic1.zhimg.com/80/v2-0e4d2f5454a69263c12df1bda8048dfa_1440w.png)

而maximum entropy reinforcement learning在**学习目标**上有所不同，除了**累积reward最大化**，还要求**policy的每一次输出action熵值最大**，也就是说**action更加随机**，不要集中在某一个action上。其优化目标公式可转化为：

![img](https://pic2.zhimg.com/80/v2-991f1623f44d71f9bd592ec5beac54d2_1440w.png)

这里的α参数用于控制用户目标是关注reward还是entropy。

### 2.4 stochastic vs deterministic policy

这个角度看，soft actor-critic算法通过**最大熵目标**学习**stochastic policies**。这里的**entropy在policy和value function中都存在**

（与A3C不同，只在训练policy中增加了entropy项，是作为regularizer项，使得policy 更加随机，训练目标不变），**加公式**

相对于deterministic，这里输出的是action的分布**一个概率分布**，本身也保证了一定的随机性。**（1）**add entropy to policy 可输出更多近似最优的行为，阻止policy variance过早收敛。**（2）**add entropy to value function是通过增加**状态空间**的value region鼓励exploration，使得action分布更加均匀。

## 3.From Soft Policy Iteration to Soft Actor-Critic

该算法可以说是从一种基于**最大熵的policy iteration 方法演化而来**。文章先是做了==理论推导==，验证算法的收敛性==，然后做了该算法的详细说明。

### 3.1 soft policy iteration

首先给出在tabular情况下收敛性的推导论证（推导细节见附录，感兴趣的可以看看）。主要的结论是：

![img](https://pic4.zhimg.com/80/v2-a71c54d1b2955b1cf11e099f65c823f5_1440w.jpg)

其中，v(st)是soft state value function, ![[公式]](https://www.zhihu.com/equation?tex=Q%5E%7Bk%2B1%7D%3DT%5E%5Cpi+Q%5Ek)

**paper中用了两个Lemma和1个theorem来说明整个过程**。

**Lemma1**(**soft policy evaluation**):主要是说明当k->无穷时，Qk会收敛到policy的soft Q函数。

**Lemma1**(**soft policy improvement**):通过如下优化方式，![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctext%7Bnew%7D%7D)不会比![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_%7B%5Ctext%7Bold%7D%7D)差，policy会得到持续改进，公式如：

![img](https://pic3.zhimg.com/80/v2-bfc52af0cf29f3cc94aa1806c36decfd_1440w.jpg)

**Theorem1**（**soft policy Iteration**）：迭代上述soft policy evaluation和soft policy improvement), 最终policy会收敛到最优。

### 3.2 Soft Actor-Critic

基于上面的理论推导，可以证明Soft policy Iteration在tabular情况下(**finite horizon**)是收敛的，用神经网络近似soft q-function和policy也能够获得较好的收敛性质，而且NN可以解决tabular难以解决的高维问题）。文中将state value function（ ![[公式]](https://www.zhihu.com/equation?tex=V_%5Cpsi%28S_t%29) ），soft Q-function Q（ ![[公式]](https://www.zhihu.com/equation?tex=Q_%5Ctheta%28S_t%2Ca_t%29) ）和a tractable policy（ ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_%5Cphi%28S_t%2C+a_t%29) ）用网络学习，一个网络（ 参数为![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) ）近似soft Q-funcion，通常是几层的MLP最后输出一个单值表示Q，用另一个网络（参数为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) ）去学policy，输出是一个分布，**一般是用高斯分布的均值和协方差表示**。**为什么用高斯分布来表示。？？？**

通常Q learning的方法是更新**Bellman residual**，而Soft Q-function的更新方法是更新Soft Bellman residual，相比于Bellman residual是增加了entropy项，具体如下：

![img](https://pic2.zhimg.com/80/v2-8af1ec073c1fdd7ca040f1e5e15b80b6_1440w.jpg)

该算法跟DDPG类似，构造了一个target soft Q网络参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpsi) 得到。

**policy网络**更新是**最小化expected KL-divergence。**

![img](https://pic3.zhimg.com/80/v2-b684780a92be942667aab157fc24bbdd_1440w.jpg)

如上所述，**policy网络需得到高斯分布的均值和协方差，但根据均值和协方差是不可导的（？？？为什么是不可导的），这里用到了一个variational autoencoder（变分自编码）里常用的reparameterization trick（重参数化技巧）**。该方式简单说来就是**不直接根据均值和协方差采样**，而是**先从一个高斯分布中采样**，然后再把**采样值乘以协方差在加上均值**。使得**网络变成可导**。用reparameterization trick后的**近似梯度**为：

![img](https://pic4.zhimg.com/80/v2-d036bd7f8755e57fa4824e370410b430_1440w.png)

## 4 算法实现

该算法的伪代码实现如下：

![img](https://picb.zhimg.com/80/v2-1e8b07dd11a24b80b03feecf25b57d13_1440w.jpg)

该算法需要**更新四组参数**，value function，target value function，soft q-function，policy。在这里还有一个trick：用两个网络去近似soft-function，然后单独通过![[公式]](https://www.zhihu.com/equation?tex=J_%7BQ%7D%28%5Ctheta_%7Bi%7D%29)优化，在优化actor（参数![[公式]](https://www.zhihu.com/equation?tex=%5Cphi)）时，使用Q值较小的网络作为critic来减少偏差，目的是为了减少policy improvement时的偏差。

随后在paper：Soft Actor-Critic Algorithms and Applications中有做了进一步的优化，其主要差别是引入了temperature ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 自动调整，在原来算法中，给了一个依赖先验的固定参数作为entropy的权重，因为reward的不断变化，采用固定的α显然是不合理的，会让整个训练不稳定，因此自动化的调整参数非常重要。那该如何设计参数α呢？总体原则是当policy探索到新的区域时，最后的action还不清楚，应该调大α，鼓励agent去探索更多的空间。当某一个区域已经探索比较充分时，最优的action已基本明确，α应该调小。

在更新的算法中，作者构造了一个带约束的优化问题，让average entropy的权重是有限制的，但在不同的state下entropy的权重是变化的。具体如下：

![img](https://pic2.zhimg.com/80/v2-5da9db65fc0009af3b97e02a92cb8480_1440w.jpg)

该算法优化后完整代码如下：

![img](https://picb.zhimg.com/80/v2-3ef8d08db662644b20f3cadd220b7d53_1440w.jpg)

可以对照代码理解两版算法。

## 5. 算法网络结构

下面是来自[nervanasystems](https://link.zhihu.com/?target=https%3A//nervanasystems.github.io/coach/components/agents/policy_optimization/sac.html)的sac网络结构图，该图是包含value network的实现。

![img](https://picb.zhimg.com/80/v2-553684ebe48ecf733f2d24379f7b4a27_1440w.jpg)

## 6. 实验

文章通过在OpenAI gym benchmark suite上几组实验验证了算法效果，以下是在文中给出的效果：

![img](https://pic2.zhimg.com/80/v2-497cd381dd445129e182c98b467f5b5a_1440w.jpg)

从实验效果看，**performance和stability**上都能取得较优的效果。具体可以看paper，在此不再赘述。

## 7. 总结

该论文的**理论思路**还是不错的，尤其对**maximum entropy的使用有一定的创新性**，且**从理论推导上较完善**，是现阶段the start-of-art的算法，不过也不少人指出文章未能像其描述一样有巨大的创新性，而且**在实现中用了较多的trick**，尽管这些**trick产生了较好的效果**。不过从实验效果看，算法效果还是很不错的，并且有较好的**stability**稳定性，目前工业界也已经有了不少应用，在一定程度上证明了算法的价值。

## 8.参考资料

1.Soft Actor-Critic:Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor

2.Soft Actor-Critic Algorithms and Applications

3.[https://zhuanlan.](https://zhuanlan.zhihu.com/p/70360272)[zhihu.com/p/70360272](http://zhihu.com/p/70360272)

4.https://zhuanlan.zhihu.com/p/52526801

## 9.code

- [https://github.com/rail-berkeley/softlearning](https://link.zhihu.com/?target=https%3A//github.com/rail-berkeley/softlearning)
- [https://github.com/vitchyr/rlkit](https://link.zhihu.com/?target=https%3A//github.com/vitchyr/rlkit)
- [https://github.com/openai/spinningup](https://link.zhihu.com/?target=https%3A//github.com/openai/spinningup)
- [https://github.com/hill-a/stable-baselines](https://link.zhihu.com/?target=https%3A//github.com/hill-a/stable-baselines)



















