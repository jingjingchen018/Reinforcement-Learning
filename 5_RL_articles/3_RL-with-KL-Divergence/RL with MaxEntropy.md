# RL with MaxEntropy

## (0) What: MaxEntropy

MaxEntropy model has been successfuly applied in Supervised Learning. 

For example, bf

Exploration entropy denotes the action selection uncertainty

##  (0) IRL (Inverse RL)

**逆强化学习的前提是需要有专家经验（expert trajectories）**



对于简单的case, 比方说最短路径问题，根据路径长短设置reward, 然后通过迭代的方式寻找到最短路径，这是RL 的思路。

但是当面对一项复杂的任务时，比方说中间遇到障碍物、交通堵塞等情况，很难给定一个具体的reward function来指引agent 进行决策。但是现实中司机面对这种情况却可以做的非常好。想找一个reward function 来指引agent(无人车)得到司机(专家)的行驶策略比较难，但是可以反过来，让agent 从司机的行为里面推导（估计、近似）出一个可以让agent收敛到司机的开车策略（policy）的reward function，这就是IRL的基本思想以及其中一个应用例子。

简单来说。RL 有环境可以去学习policy, 而IRL 是反过来学习环境，

IRL 是指在给定一个策略（optimal or not）或者一些操作示范的前提下，反向推导出MDPs的reward function，让智能体通过专家示范（**expert trajectories**）(假设是optimal or near optimal)来学习如何决策复杂问题的一种算法。我们假设专家在完成某项任务时，其决策往往是最优的或接近最优的，当所有的策略产生的累积汇报函数期望都不比专家策略产生的累积回报期望大时，强化学习所对应的回报函数就是根据示例学到的回报函数。即逆向强化学习就是从专家示例中学习回报函数。当需要基于最优序列样本学习策略时，我们可以结合逆向强化学习和强化学习共同提高回报函数的精确度和策略的效果。
链接：https://www.zhihu.com/question/68237021/answer/699161675

IRL 不用手工设计Reward，通过学习的方式learn the reward function.因为手工设计的reward function  很难考虑周全。简单点的有，假设了reward function是某个结构（例如：reward = w1*p1+w2*p2），然后学习这个结构中相应参数的权重，例如**Liner program IRL** 中就是通过**凸优化学习到w1和w2（这是很早期的算法，08年**）；后来就出现了一些新的算法，像**高斯过程IRL，不过不适用于状态规模较大的情况**，又比如**Maximum Entropy IRL（最大熵IRL）**，它用**神经网络来学习reward**，可以学习到一个非线性的function（Liner program IRL 是线性的）。除了reward function 这一点区别以外，其它沿袭了RL。

链接：https://www.zhihu.com/question/68237021/answer/398633403


The main idea of inverse reinforcement learning is to **learn the reward function** based on **the agent's decisions**, and then **find the optimal policy** using **reinforcement learning techniques.**


链接：https://www.zhihu.com/question/68237021/answer/1386348651

## (1) Why: using MaxEntropy in RL

从目前我知道的最早的文章开始，

首先是利用凸优化学习 linear reward function 线性的reward function， 再就是利用 Gauss 模型、NN学习非线性的reward function.  从单个reward function 到多个reward function （1.1-1.3） 

The commonalities of these articles (1.1-1.3) are estimating reward functions. (from linear reward function to non-linear reward function )

**Question: can they all converge to the optimal policy or suboptimal policy??  If they can, how they verify or deduction ? **

(1.1) (2007 ICJAI) Bayesian Inverse Reinforcement Learning  伊利诺伊大学厄巴纳-尚佩恩分校

Abstract:

> Inverse Reinforcement Learning (IRL) is the problem of learning the reward function underlying a Markov Decision Process given the dynamics of the system and the behaviour of an expert. IRL is motivated by situations where knowledge of the rewards is a goal by itself (as in preference elicitation) and by the task of apprenticeship learning (learning policies from an expert). In this paper we show **how to combine prior knowledge** and **evidence from the expert’s actions** to derive a **probability distribution** over **the space of reward functions**. We present efficient algorithms that find solutions for the reward learning and apprenticeship learning tasks that generalize well over these distributions. Experimental results show strong improvement for our methods over previous heuristic-based approaches.

```
@inproceedings{ramachandran2007bayesian,
  title={Bayesian Inverse Reinforcement Learning.},
  author={Ramachandran, Deepak and Amir, Eyal},
  booktitle={IJCAI},
  volume={7},
  pages={2586--2591},
  year={2007}
}
```

Technical details:



(1.2) (2012 NIPS) Nonparametric Bayesian Inverse Reinforcement Learning for **Multiple Reward Functions**. 这两篇文章都是一个韩国人Kim, Kee-Eung  

Code:http://ailab.kaist.ac.kr/codes/nonparametric-bayesian-inverse-reinforcement-learning-for-multiple-reward-functions

```
@inproceedings{choi2012nonparametric,
  title={Nonparametric Bayesian inverse reinforcement learning for multiple reward functions},
  author={Choi, Jaedeug and Kim, Kee-Eung},
  booktitle={Advances in Neural Information Processing Systems},
  pages={305--313},
  year={2012}
}
```

Abstract：

> We present a nonparametric Bayesian approach to inverse reinforcement learning (IRL) for multiple reward functions. Most previous IRL algorithms assume that the behaviour data is obtained from an agent who is optimizing a **single reward **function, but this assumption is hard to be met in practice. Our approach is based on **integrating the Dirichlet process mixture model** into **Bayesian IRL**. We provide an efficient **Metropolis-Hastings sampling algorithm** utilizing the **gradient of the posterior** to **estimate the underlying reward functions**, and demonstrate that our approach outperforms the previous ones via experiments on a number of problem domains.

Technical details:



Or theorical details:



(1.3) (2011 NIPS ) MAP Inference for Bayesian Inverse Reinforcement Learning

> The difficulty in inverse reinforcement learning (IRL) arises in choosing the best reward function since there are typically an infinite number of reward functions that yield the given behaviour data as optimal. Using a Bayesian framework, we address this challenge by using the maximum a posteriori (MAP) estimation for the reward function, and show that most of the previous IRL algorithms can be modeled into our framework. We also present a gradient method for the MAP estimation based on the (sub)differentiability of the posterior distribution. We show the effectiveness of our approach by comparing the performance of the proposed method to those of the previous algorithms.
>
> 

Technical details:



Or theorical details:



(1.4) Maximum Entropy was first used in Inverse RL (IRL).  **2008**

目的是为了学习reward function

出自卡耐基梅隆大学

https://zhuanlan.zhihu.com/p/91819689

https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf?source=post_page---------------------------

*Maximum entropy inverse reinforcement learning*

逆强化学习要解决的关键问题是：根据专家数据倒推reward function 

Assumptions:

(a) State space, Action space

(b) Roll-outs from $\pi^{*}$ (what's the )

Objective:

(a)还原回报函数   the learned reward function 


$$
R_{\varphi}(\tau)=\sum_{t}r_{\varphi}(s_{t},a_{t})
$$
where $\varphi$ are learned parameters, $\tau=\left\{s_{1},a_{1},s_{2},a_{2},\ldots, s_{t},a_{t},\ldots,s_{T}\right\}$ is the trajectory. 



(b) 根据回报函数进行策略寻优expert demonstrations：
$$
\left\{\tau_{i}\right\}\sim \pi^{*}
$$
Policy $\tau$的reward 越大，则表示$\tau$应该出现的几率越大
$$
p(\tau)=\frac{exp(R_{\varphi}(\tau))}{\int R_{\varphi}(\tau)d\tau}
$$


To infer the reward function, we **maximize the log likelihood** of **our set of demonstrations** with respect to the**parameters** of our reward function.
$$
\max _{\psi} \sum_{\tau \in \mathcal{D}} \log p_{r_{\psi}}(\tau)
$$
同样的，这也是一个优化问题，可以用gradient descent method 来求解。Abstract：

> Recent research has shown the benefit of framing problems of imitation learning as solutions to Markov Decision Problems. This approach reduces learning to the problem of recovering a utility function that makes the behavior induced by a near-optimal policy closely mimic demonstrated behavior. In this work, we develop a **probabilistic approach** based on the **principle of maximum entropy**. Our approach provides a well-defined, globally normalized distribution over decision sequences, while providing the same performance guarantees as existing methods. We develop our technique in the context of modeling realworld navigation and driving behaviors where collected data is inherently noisy and imperfect. Our probabilistic approach enables modeling of **route preferences** as well as a powerful new approach to inferring destinations and routes based on partial trajectories.

```
@inproceedings{ziebart2008maximum,
  title={Maximum entropy inverse reinforcement learning.},
  author={Ziebart, Brian D and Maas, Andrew L and Bagnell, J Andrew and Dey, Anind K},
  booktitle={Aaai},
  volume={8},
  pages={1433--1438},
  year={2008},
  organization={Chicago, IL, USA}
}
```

Comments:

Technical details:

Or theorical details:

**Advantage:**



**Disadvantage:**

The algorithm is only suitable for **low dimension state space**, **action space and state transaction probability** is needed.  (是否可以通过MDP分解进行求解大的状态空间，divided and conquer.????这里的意思是说低维的状态空间，和状态空间的大小没有关系，所以这里的问题是低维的状态空间是指的小的状态空间吗？) 



可以把参数化的MDP 这个问题，放松条件限制，到局部MDP的问题中么



*(1.5)（2020ICLR）If MaxEnt RL is the Answer, What is the Question*

出自卡耐基梅隆, UC 伯克利 google brain, 

> @article{eysenbach2019if,
>   title={If MaxEnt RL is the Answer, What is the Question?},
>   author={Eysenbach, Benjamin and Levine, Sergey},
>   journal={arXiv preprint arXiv:1910.01913},
>   year={2019}
> }

Abstract:

> 

Comments:



Technical details:



Or theorical details:



**Advantage:**



**Disadvantage:**



# # Why:

## Exploration:

Probabilistic Inference

## easiler optimization





## (2) How: Three Scenes

$$
\theta_{t+1}=\theta_{t}+\alpha\nabla J(\theta)
$$

Step size $\alpha$ will effect the convergence of the parameter. 

To solve this question,  researchers 

**Actually， there are no algorithms related to entropy  applied in solving COP**

### (2.1) As a regularization



 ####  (a) TRPO

using  surrogate function to approximate true objective functions.

There are four tricks in TRPO: 



Objective:
$$
\begin{aligned}
&\operatorname{maximize}_{\theta} E_{s} \pi_{\theta_{o l d}}, a\pi_{\theta_{o l d}}\left[\frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{o l d}}(a \mid s)} A_{\theta_{o l d}}(s, a)\right]\\
&\text {subject to } E_{s} \pi_{\theta_{\text {old}}}\left[D_{K L}\left(\pi_{\theta_{\text {old}}}(\cdot \mid s) \| \pi_{\theta}(\cdot \mid s)\right)\right] \leq \delta
\end{aligned}
$$


Advantage：



#### (b) PPO

**Commonality**



### (2.2) As objective,



#### (a) Soft AC

State-of-art 

Objective:
$$
J(\pi)=\sum_{t=0}^{\infty} \mathbb{E}_{\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right) \sim \rho_{\pi}}\left[\sum_{l=t}^{\infty} \gamma^{l-t} \mathbb{E}_{\mathbf{s}_{l} \sim p, \mathbf{a}_{l} \sim \pi}\left[r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)+\alpha \mathcal{H}\left(\pi\left(\cdot \mid \mathbf{s}_{t}\right)\right) \mid \mathbf{s}_{t}, \mathbf{a}_{t}\right]\right]
$$
Where 
$$
\mathcal{H}\left(\pi\left(\cdot \mid \mathbf{s}_{t}\right)\right)=
$$


Advantage:



#### (b) Soft Q-learning

https://arxiv.org/pdf/1702.08165.pdf

> ```
> @article{haarnoja2017reinforcement,
>   title={Reinforcement learning with deep energy-based policies},
>   author={Haarnoja, Tuomas and Tang, Haoran and Abbeel, Pieter and Levine, Sergey},
>   journal={arXiv preprint arXiv:1702.08165},
>   year={2017}
> }
> ```

Objective:
$$
\begin{array}{l}
J_{\pi}\left(\phi ; \mathbf{s}_{t}\right)= \\
\mathrm{D}_{\mathrm{KL}}\left(\pi^{\phi}\left(\cdot \mid \mathbf{s}_{t}\right) \| \exp \left(\frac{1}{\alpha}\left(Q_{\mathrm{soft}}^{\theta}\left(\mathbf{s}_{t}, \cdot\right)-V_{\mathrm{soft}}^{\theta}\right)\right)\right)
\end{array}
$$



$$
\frac{\partial J_{\pi}\left(\phi ; \mathbf{s}_{t}\right)}{\partial \phi} \propto \mathbb{E}_{\xi}\left[\Delta f^{\phi}\left(\xi ; \mathbf{s}_{t}\right) \frac{\partial f^{\phi}\left(\xi ; \mathbf{s}_{t}\right)}{\partial \phi}\right]
$$


**Commonality**



### (c) Divided and Conquer RL 

Motivation:





Objective:
$$
\mathcal{L}_{\text {center}}\left(\pi_{c}\right)=\mathbb{E}_{\pi}\left[D_{K L}\left(\pi(\cdot \mid s) \| \pi_{c}(\cdot \mid s)\right)\right] \propto \sum_{i} \rho\left(\omega_{i}\right) \mathbb{E}_{\pi_{i}}\left[-\log \pi_{c}(s, a)\right]
$$




### (2.3) Using in MARL

#### (a) Wang Jun and Wen Ying et.al----applying MaxEntropy to MARL 

*P1 （ICJAI2019）A Regularized Opponent Model with Maximum Entropy Objective*

> @article{tian2019regularized,
>   title={A regularized opponent model with maximum entropy objective},
>   author={Tian, Zheng and Wen, Ying and Gong, Zhichen and Punakkath, Faiz and Zou, Shihao and Wang, Jun},
>   journal={arXiv preprint arXiv:1905.08087},
>   year={2019}
> }

*P2 Probabilistic Recursive Reasoning for Multi-Agent Reinforcement Learning*

## (3) 

| Author | Title | Objective |      |
| ------ | ----- | --------- | ---- |
|        |       |           |      |
|        |       |           |      |
|        |       |           |      |
|        |       |           |      |



























