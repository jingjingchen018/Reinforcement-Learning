### Slide 1 **Conditional Probability**

events: 

a simple example:



why it's so important?

In Machine Learning,  e.g. MLE 



### Slide 2 The law of total expectation



a simple example:



why it's so important?





### Slide 3 Markov Property

马尔科夫性指的是系统的下一个状态$S_{t+1}$仅与当前状态$S_{t}$ 有关，而与以前的状态无关。

Definition1:

 状态$S_{t}$是马尔科夫的，当且仅当
$$
P[S_{t+1}|S_{t}]=P[S_{t+1}|S_{1},S_{2},\ldots,S_{t}]
$$
从定义中我们可以看到，状态$S_{t}$ 蕴含了所有相关的历史信息$S_{1},S_{2},\ldots,S_{t}$, 一旦已知当前状态$S_{t}$,前面的历史信息将会被抛弃。（因为蕴含在状态$S_{t}$中了).

Definition2: Markov Chain

状态空间$S$中，状态序列${S_{t}}$是一个马尔可夫链，对于任意的状态 都满足马尔科夫性。
$$
P[S_{t+1}=s|S_{t}]=P[S_{t+1}=s|S_{1},S_{2},\ldots,S_{t}]
$$
然而马尔科夫性描述的只是单个状态的性质。

对于一个状态序列，数学中有专门用来描述随机变量序列的学科，即随机过程。 随机过程就是随机变量序列。如果这个随机变量序列中的每个状态都是马尔科夫的，那么，这个随机过程就叫做马尔科夫随机过程。又叫做Markov Process

Definition3:

Markov Process是一个二元组$(S,P)$, 其中$S$ 是有限状态集合，$P$是状态转移概率矩阵。
$$
P=\begin{bmatrix}
P_{11} & \cdots & P_{1n}\\
\vdots & \ddots & \vdots \\
P_{n1} & \cdots & P_{nn}
\end{bmatrix}
$$


![image-20201015145724448](/Users/chenjingjing/Library/Application Support/typora-user-images/image-20201015145724448.png)

如图1.2所示为一个学生的7种状态{娱乐，课程1，课程2， 课程3，考过，睡觉，论文}，每种状态之间的转移概率如图所知。则该生从课程1开始一天可能的状态序列为：

课1-课2-课3-考过-睡觉

课1-课2-睡觉

这样的状态序列称为马尔可夫链。



当状态转移概率给定的时候，从某个状态$S_{0}$出发,会有多条马尔科夫链。 在机器人或者游戏领域中，也就是强化学习领域中，马尔可夫决策过程不足以描述其性质，因为这些应用都涉及到通过动作$A_{t}$与环境进行交互，从而获得奖励$R_{t}$。将动作和回报考虑在内的Markov Process称作Markov Decision Process.

### Slide 4 Markov Decision Process.

**Definition3: Markov Decision Process.**

马尔科夫决策过程由一个五元组组成 $(S,A,P,R,\gamma)$. 其中 

1. $S$ is  the state space,

2. $A$ is the action space, 

3.  $P$ is the transition probability, $P:S\times A \times S\rightarrow R$ 
     $P(s^\prime|s,a)$ is the probability of transitioning into state $s^\prime$ upon executing action $a$ in state $s$, 

4. $r$ is the reward function, $r:S\times A \rightarrow R$  $r(s,a)$ is the immediate reward associated with taking action $a$ in state $s$,  

5. $\gamma$ is the discount factor for future reward, where $\gamma \in [0,1]$. 用来计算累计回报。

   Different from Markov Process, transition probability in MDP includes action :
   $$
   P^{a}_{s,s^{\prime}}=P[S_{t+1}=s^{\prime}|S_{t}=s,A_{t}=a]
   $$
   $S_{t}, A_{t}$ 都是随机变量(大写表示随机变量，小写表示取值) , $P^{a}_{ss^{\prime}}$ 表示的是在t时刻从状态$s$，采取动作$a$，在下一个时刻转移到状态$s^{\prime}$的概率。

    Note: 

a simple example:



![image-20201015154613414](/Users/chenjingjing/Library/Application Support/typora-user-images/image-20201015154613414.png)

学生有五个状态，状态集$S=\left\{s_{1},s_{2},s_{3},s_{4},s_{5}\right\}$,动作集为$A=\left\{玩，退出，学习，发论文，睡觉\right\}$, immediate reward 用红色标记。$R=r(s,a)$

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

**符号的归纳与统一**

很多情况下，不同的教材用的符号有区别，在这里特别强调下。

In Sutton

Given any state $s$ and action $a$, the probability of each possible pair of next state and reward, $s^{\prime}$, $r$, is denoted 
$$
p(s^{\prime},r|s,a)=Pr\left\{S_{t+1}=s^{\prime},R_{t+1}=r|S_{t}=s,A_{t}=a\right\}
$$
Given the dynamics as specified by 上式， one can compute anything else one might want to know about the environment, such as the expected rewards for state-action pairs. 
$$
r(s,a)=\mathrm{E}[R_{t+1}|S_{t}=s,A_{t}=a]=\sum_{r\in
\mathcal{R}}r\sum_{s^{\prime}\in S}p(s^{\prime},r|s,a)
$$
这里的回报的意思是，采取状态行为对$(s,a)$之后得到reward的期望值。 

the state-transition probabilities,
$$
p\left(s^{\prime} \mid s, a\right)=\operatorname{Pr}\left\{S_{t+1}=s^{\prime} \mid S_{t}=s, A_{t}=a\right\}=\sum_{r \in \mathcal{R}} p\left(s^{\prime}, r \mid s, a\right)
$$
这里是表示，采取状态行为对$(s,a)$之后
$$
r\left(s, a, s^{\prime}\right)=\mathbb{E}\left[R_{t+1} \mid S_{t}=s, A_{t}=a, S_{t+1}=s^{\prime}\right]=\frac{\sum_{r \in \mathcal{R}} r p\left(s^{\prime}, r \mid s, a\right)}{p\left(s^{\prime} \mid s, a\right)}
$$
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

由上面两个式子可以知道
$$
\sum_{s^{\prime}}[p(s^{\prime}|s,a)r(s,a,s^{\prime})]=
\sum_{r\in \mathcal{R}}r\sum_{s^{\prime}}p(s^{\prime},r|s,a)=r(s,a)
$$


但其实没有那么复杂，在大多数书籍和文章中，并没有使用这一套定义。对于instant reward的定义就是采取状态行为对$(s,a)$后的回报。
$$
r(s,a,s')=R^{a}_{s,s^{\prime}}\\
r(s,a)=\sum_{s^{\prime}}p(s^{\prime}|s,a)r(s,a,s^{\prime})\\
$$
Note: 上式不一定准确，（所以我还是有一点点迷惑$R_{t+1}=r(s_{t},a_{t})$）

在deterministic  policy 下，$r(s,a)不一定=r(s,a,s^{\prime})$

why it's so important?

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

强化学习的目标在于，对于给定的MDP, 寻找一个optimal policy。

而policy (decision rule) 是指的是从状态到动作的映射？？？通常用符号$\pi$表示。

(1) $\pi(s)$: action taken in state $s$ under deterministic policy $\pi$.

(2) $\pi(a|s)=p(A_{t}=a|S_{t}=s)$: probability of taking action $a$ in state $s$ under stochastic policy $\pi$.

它表示给定状态$s$时，动作集上的一个分布，这是一个条件概率。



Note: 强化学习中采取更多的是随机策略。因为随机不会带来更大的收益（in math），但是会带来更多的随机性，也就是exploration。通过随机采样的方式，agent(e.g. 机器人)可以通过探索找到更好的policy. 

*而在实际应用中，存在各种噪声，这些噪声大多都服从正态分布，如何去掉这些噪声，也需要用到概率的知识。*

（2）中的公式的含义是，策略$\pi$在每个状态指定一个动作概率，如果给出的策略$\pi$是确定性的，那么策略$\pi$ 在每个状态$s$指定一个确定的动作。

言归正传，公式(1.1)的含义是：策略![[公式]](https://www.zhihu.com/equation?tex=%5Cpi)在每个状态![[公式]](https://www.zhihu.com/equation?tex=s) 指定一个动作概率。如果给出的策略![[公式]](https://www.zhihu.com/equation?tex=%5Cpi)是确定性的，那么策略![[公式]](https://www.zhihu.com/equation?tex=%5Cpi)在每个状态![[公式]](https://www.zhihu.com/equation?tex=s)指定一个确定的动作。

> 例如其中一个学生的策略为![[公式]](https://www.zhihu.com/equation?tex=%5C%5B+%5Cpi_1%5Cleft%28%5Ctextrm%7B%E7%8E%A9%7D%7Cs_1%5Cright%29%3D0.8+%5C%5D)，是指该学生在状态![[公式]](https://www.zhihu.com/equation?tex=s_1) 时玩的概率为0.8，不玩的概率是0.2，显然这个学生更喜欢玩。
>
> 另外一个学生的策略为![[公式]](https://www.zhihu.com/equation?tex=%5Cpi_2%5Cleft%28%5Ctextrm%7B%E7%8E%A9%7D%7Cs_1%5Cright%29%3D0.3)，是指该学生在状态![[公式]](https://www.zhihu.com/equation?tex=s_1)时玩的概率是0.3，显然这个学生不爱玩。依此类推，每学生都有自己的策略。强化学习是找到最优的策略，这里的最优是指得到的总回报最大。



给定policy $\pi$, 从起始状态$S_{1}$出发， 我们可以获得多条trajectory,

$\tau=\left\{S_{1},S_{2},\ldots\right\}$

这是一个随机变量序列。 (一般这里还没有考虑到初始状态$s$的分布$\rho_{0}$.)

假如从$t$时刻开始，计算累计回报，从状态$S_{t}$出发，他的累积回报就是 
$$
G_{t}=R_{t+1}+\gamma R_{t+2} + \gamma^{2} R_{t+3}+ \cdots =\sum_{k=0}^{\infty}\gamma^{k} R_{t+k+1}
$$
因为策略$\pi$是随机的，Reward$R_{t}$, (cumulative discounted rewards)累计回报$G_{t}$（returns）也是一个随机变量。它可以根据状态序列的不同，得到的值也不一样。

**那我们自然的一个想法就是通过一个确定的值，来描述起始状态$S_{t}=s$的价值. 期望是一个确定值，可以作为状态值函数的定义**

Definition 5:

当智能体采用策略$\pi$ 时，累计回报$G_{t}$服从一个分布，将累计回报在状态$s$处的期望值定义为状态值函数。
$$
V_{\pi}(s)  =\mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty}\gamma^{k} R_{t+k+1}|S_{t}=s\right]\\
=\mathbb{E}_{\pi}\left[G_{t} | S_{t}=s\right]\\
  =\mathbb{E}_{\pi}\left[R_{t+1}+\gamma G_{t+1} | S_{t}=s\right]\\
$$
Note: 状态值函数和策略$\pi$  是相对应的，这是因为策略$\pi$ 决定了累计回报$G$的分布。所以，期望的下标可以是$\pi$，也可以是轨迹$\mathrm{E}_{S_{t},S_{t+1},\cdots}[G_{t}|S_{t}=s]$

%%%%%%%%%%%%%%%

In MDPs,  there always exists a deterministic stationary policy (that simultaneously maximizes the value of every state) 

%%%%%%%%%%%%%%%

 Definition6: 

相应的，状态-行为值函数的定义为
$$
q_{\pi}(s,a)  =\mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty}\gamma^{k} R_{t+k+1}|S_{t}=s,A_{t}=a\right]\\
=\mathbb{E}_{\pi}\left[G_{t} | S_{t}=s,A_{t}=a\right]\\
  =\mathbb{E}_{\pi}\left[R_{t+1}+\gamma G_{t+1} | S_{t}=s,A_{t}=a\right]\\
$$
但是这两个定义式给出了状态值函数和状态行为值函数的定义计算式，在实际真正计算的时候，并不会按照定义式去编程。

### Slide 5 V(s) 和Q(s,a) 的Bellman 方程

*为什么要用bellman equation?? 这个就是直接的贝尔曼方程么？*

**state value function**
$$
V_{\pi}(s)  =\mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty}\gamma^{k} R_{t+k+1}|S_{t}=s\right]\\
=\mathbb{E}_{\pi}\left[G_{t} | S_{t}=s\right]\\
  =\mathbb{E}_{\pi}\left[R_{t+1}+\gamma G_{t+1} | S_{t}=s\right]\\
  =\mathbb{E}_{S_{t},S_{t+1},\cdots}\left[R_{t+1}+\gamma G_{t+1} | S_{t}=s\right]\\
这里不要（a）也行
 \stackrel{(a)}=\mathbb{E}_{S_{t}}[R_{t+1}+\gamma \mathbb{E}_{S_{t+1},\cdots}[G_{t+1}] | S_{t}=s]\\
  \stackrel{(b)}=\mathbb{E}_{S_{t}}[R_{t+1}+\gamma \mathbb{E}_{S_{t+1},\cdots}[G_{t+1}|S_{t+1}] | S_{t}=s]\\
\stackrel{(c)}=\mathbb{E}_{S_{t}}[R_{t+1}+\gamma V_{\pi}(S_{t+1})|S_{t}=s]\\
    \stackrel{(d)}=\mathbb{E}_{\pi}[R_{t+1}+\gamma V_{\pi}(S_{t+1})|S_{t}=s]\\ 按照条件期望公式展开,\pi 决定了在状态S_{t}要采取的动作\\
        \stackrel{(e)}=\sum_{a\in A}\pi(a|s)[r(s,a)+\gamma \sum_{s^{\prime}}P(s^{\prime}|s,a)V_{\pi}(s^{\prime})]\\
       \stackrel{(f)}= \sum_{a\in A}\pi(a|s)r(s,a)+\gamma \sum_{a\in A}\pi(a|s)\sum_{s^{\prime}}P(s^{\prime}|s,a)V_{\pi}(s^{\prime})\\
       \stackrel{(g交换变量求和顺序)}= \sum_{a\in A}\pi(a|s)r(s,a)+\gamma \sum_{s^{\prime}} \sum_{a\in A}\pi(a|s)P(s^{\prime}|s,a)V_{\pi}(s^{\prime})\\
       \stackrel{(h)}= r(s,\pi(s))+\gamma \sum_{s^{\prime}} P(s^{\prime}|s)V_{\pi}(s^{\prime})\\
         \stackrel{(i)}= r(s,\pi(s))+\gamma \sum_{s^{\prime}} P(s^{\prime}|s,\pi(s))V_{\pi}(s^{\prime})\\
       这里的r(s,\pi(s)) 既可以是deterministic,也可stochastic的回报\\
        写成向量的形式\\        
        V=R+\gamma PV\\
        V=(I-\gamma P)^{-1}R
$$
**Note:** (a) 注意是对哪些变量求期望

​			(b) 从$t+1$时刻开始计算期望。所以这里有一个变量$S_{t+1}$,这里sutton 的书里面有说过，不是完全的conditional probability.但是又有相似之处，因为$G_{t}$的计算，本身就是基于随机变量$S_{t}$的。同理计算$G_{t+1}$，就是是基于随机变量$S_{t+1}$的。这里不是很严格。

​			(c) 利用$V_{\pi}$的定义.

​			(d) 期望的脚标可以改成 $\pi$。

 $ (\mathbb{E}[\mathbb{E}[S|N]]=\mathbb{E}[S])$

**state-action value function**

同理，我们可以得到**state-action value function** 的bellman equation
$$
q_{\pi}(s,a)  =\mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty}\gamma^{k} R_{t+k+1}|S_{t}=s,A_{t}=a\right]\\
=\mathbb{E}_{\pi}[R_{t+1}+\gamma q_{\pi}(S_{t+1},A_{t+1})|S_{t}=s,A_{t}=a]\\
$$
状态值函数和状态-动作值函数的具体计算过程。其中空心圆表示状态，实心圆表示state-action pair。

*可以举给植物浇水或者缺水的例子*

我们首先来看状态行为值函数和动作行为值函数的关系，
$$
按照图示计算\\
V_{\pi}(s) = \sum_{a\in A}\pi(a|s)q_{\pi}(s, a)
$$
State-action value function 
$$
q_{\pi}(s, a) =r(s, a)+\gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \mid s, a\right) V^{\pi}\left(s^{\prime}\right)\\
  =R_{s}^{a}+\gamma \sum_{s^{\prime} \in S} P_{ss^{\prime}}^{a} V_{\pi}\left(s^{\prime}\right)\\
$$
![preview](https://pic1.zhimg.com/v2-028e973c94f1e57babf693cd05392d0c_r.jpg)

将上述两个式子结合到一块，得到state-value function
$$
V_{\pi}(s) = \sum_{a\in A}\pi(a|s)[R_{s}^{a}+\gamma \sum_{s^{\prime} \in S} P_{ss^{\prime}}^{a} V_{\pi}\left(s^{\prime}\right)]
$$
![img](https://pic1.zhimg.com/80/v2-a74a8efcf3199cef76b7e5d41b580934_1440w.jpg)

State-action value function的求解：
$$
V_{\pi}(s^{\prime}) = \sum_{a^{\prime}\in A}\pi(a^{\prime}|s^{\prime})q_{\pi}(s^{\prime}, a^{\prime})
$$
同理，得到State-action value function
$$
q_{\pi}(s, a)=R_{s}^{a}+\gamma \sum_{s^{\prime} \in S} P_{ss^{\prime}}^{a} \sum_{a^{\prime}\in A}\pi(a^{\prime}|s^{\prime})q_{\pi}(s^{\prime}, a^{\prime})\\
$$
计算状态值函数的目的是为了构建学习算法从数据中得到optimal policy. 每个policy 对应着一个state value function, optimal policy 对应着最优的state-value function

### Slide 6 Optimal value function

最优的值函数为在所有策略对应的值函数中取值最大的值函数。
$$
V^{*}(s)  = \max_{\pi}V_{\pi}(s)\\
=\max_{a}[R_{s}^{a}+\gamma \sum_{s^{\prime} \in S} P_{ss^{\prime}}^{a} V^{*}\left(s^{\prime}\right)]\\=\max_{a}\mathbb{E}[r.v.]
$$
最优的state-action value function 为在所有策略对应的状态行为值函数中取值最大的状态行为值函数。
$$
q^{*}(s, a) =\max_{\pi} q_{\pi}(s,a)\\=R_{s}^{a}+\gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \mid s, a\right)\max_{a^{\prime}} q^{*}\left(s^{\prime},a^{\prime}\right)\\=\mathbb{E}[\max(r.v.)]
$$
若已知optimal state-action value function, we can obtain optimal policy by maximizing $q^{*}(s, a)$ directly.
$$
\begin{equation}
\pi_{*}(a \mid s)=\left\{\begin{array}{c}
& 1 \quad if \quad a=\arg \max _{a \in A} q_{*}(s, a) \\
& 0 \text { otherwise}
\end{array}\right.
\end{equation}
$$
for example，当给定策略 $\pi$时，假设从状态$s_{1}$出发，那么学生的状态序列可能有如下的一些情况。
$$
s_{1}\rightarrow s_{2}\rightarrow s_{3}\rightarrow s_{4}\rightarrow s_{5};\\
s_{1}\rightarrow s_{2}\rightarrow s_{3}\rightarrow s_{5}\\
\cdots\cdots
$$
那么利用计算累计回报的公式，计算$G_{1}$, 这时$G_{1}$就有多个可能的值。

![image-20201015203535936](/Users/chenjingjing/Library/Application Support/typora-user-images/image-20201015203535936.png)

对于这个例子，状态值函数的计算。首先假设
$$
v_{\pi}(s_{1})=-1+0.5
$$




### Slide 8

(1) 对于离散时间有限范围的带折扣的MDP$M=(S,A,P,r,\gamma,\rho_{0},T)$

随机策略$\pi$下对应多条trajectory $\tau=(s_{0},a_{0},s_{1},a_{1},...)$, 这里应该是$\tau=(s_{0},a_{0},r_{1},s_{1},a_{1},r_{2}...)$, 累计回报为$R=\sum_{t=0}^{T}\gamma^{t}r_{t}$:

强化学习的目标就是，找到最优的policy, 使得该策略下，累计回报的期望最大。

The agent's goal is to maximize the cumulative reward it recieves in the long run.
$$
\max_{\pi}\int R(\tau)p_{\pi}(\tau)d\tau\\
=\max_{\pi}\mathrm{E}_{\tau \sim \rho(\tau)}[\sum_{t=0}^{T}\gamma^{t}r(s_{t},a_{t})]
$$

### Slide 7  policy gradient method

直接对policy 进行参数化
$$
V_{\pi}(s) = \sum_{a\in A}\pi(a|s;\theta)q_{\pi}(s, a)
$$
张志华是对这个公式进行求导的。最后得出REINFORCE 法。

### Slide 8 机器学习

**Objectice: Loss Function**
$$
\min \mathrm{E}(g(x,\theta))
$$

$$
\nabla g(x,\theta)=0
$$

更新公式
$$
\theta_{n+1}=\theta_{n}+\alpha_{n}\nabla g(x_{n+1},\theta_{n})
$$
在机器学习中，牛顿法、最速下降法，等等迭代公式都是这样更新参数的。

### Slide 9 RL REINFPRCE   算法。

在这里policy 是被参数化的，对于参数化的policy $\pi_{\theta}$, 它的trajectory $\tau=(s_{1},a_{1},s_{2},a_{2},...)$, 累计回报为$R(\tau)=\sum_{t=1}^{T}\gamma^{t}r(s_{t},a_{t})$

(这里也应该是用大写来表示随机变量)

参考的推导过程是累计回报为$R(\tau)=\sum_{t=1}^{T}\gamma^{t}r(s_{t},a_{t},s_{t+1})$

所以都是默认deterministic policy?????????????????

$\pi_{\theta}$ 相应的期望值就是
$$
J(\theta):=\mathrm{E}_{\tau \sim p(\tau; \theta)}[\sum_{t=1}^{T}\gamma^{t}r(s_{t},a_{t})]\\
=\int R(\tau)p(\tau;\theta)d\tau \tag{9.2}
$$
又 因为轨迹的markov性，那么轨迹的概率为
$$
p(\tau;\theta)=p(s_{1})\prod_{t=1}^{T-1}p(s_{t+1}|s_{t},a_{t})\pi(a_{t}|s_{t};\theta)
$$
要求optimal policy,即求最优的参数$\theta^{*}$
$$
\theta^{*}=\arg\max_{\theta}J(\theta)
$$
但是$J(\theta)$并不好求，这时候我们通过对其求导来更新参数
$$
\theta_{n+1}=\theta_{n}+\alpha\nabla J(\theta_{n})
$$
其中，（这里也可以写成求和公式），
$$
\nabla J(\theta)=\nabla_{\theta} \int R(\tau)p(\tau;\theta)d\tau \\
=\int R(\tau)\nabla_{\theta} p(\tau;\theta)d\tau \\
=\int R(\tau)p(\tau;\theta)\frac{\nabla_{\theta} p(\tau;\theta)}{p(\tau;\theta)}d\tau \\
= \int p(\tau;\theta)R(\tau)\nabla_{\theta}\ln p(\tau;\theta)d\tau \\
= \mathrm{E}_{\tau \sim p(\tau;\theta)}[R(\tau)\nabla_{\theta}\ln p(\tau;\theta)] \\
通过经验平均估计(即通过*采样*m条轨迹后，去计算策略梯度)\\
把轨迹的概率公式展开\\
= \mathrm{E}_{\tau\sim p(\tau;\theta)}[R(\tau)\nabla_{\theta}\ln p(\tau;\theta)] \\
p(s_{1})\prod_{t=1}^{T-1}p(s_{t+1}|s_{t},a_{t})\pi(a_{t}|s_{t};\theta) 求ln之后再对
\theta 求导为0  \\
= \mathrm{E}_{\tau \sim p(\tau;\theta)}[R(\tau) \sum_{t=1}^{T-1} \nabla_{\theta} \ln \pi(a_{t}|s_{t};\theta)] \\
$$
因为轨迹的分布$p(\tau;\theta)$是未知的，所以可以使用经验平均**找一个无偏估计**的方法去近似$\nabla J(\theta)$
$$
\nabla \hat{J}(\theta)=\frac{1}{m}\sum_{n=1}^{m}\sum_{t=1}^{T-1}[R(\tau_{n}) \sum_{t=1}^{T-1} \nabla_{\theta} \ln \pi(a_{t,n}|s_{t,n};\theta)]
$$
 其中$\tau_{n}=(s_{1,n},a_{1,n},s_{2,n},a_{2,n},...)$

### Slide 10 采样方法

MC 采样

importance sampling 

recieve- reject sampling

### Slide 11 Baseline REINFORCE Variance=0



**Conditional Probability**

a simple example:



why it's so important?

In Machine Learning, e.g. MLE

# Slide 11

**Conditional Probability**

a simple example:



why it's so important?

In Machine Learning, e.g. MLE

### Slide 12

**Conditional Probability**

a simple example:



why it's so important?

In Machine Learning, e.g. MLE

### Slide 13

**Conditional Probability**

a simple example:



why it's so important?

In Machine Learning, e.g. MLE

### Slide 14

**Conditional Probability**

a simple example:



why it's so important?

In Machine Learning, e.g. MLE

### Slide 15

**Conditional Probability**

a simple example:



why it's so important?

In Machine Learning, e.g. MLE

### Slide 16

**Conditional Probability**

a simple example:



why it's so important?

In Machine Learning, e.g. MLE

### Slide 17

**Conditional Probability**

a simple example:



why it's so important?

In Machine Learning, e.g. MLE

### Slide 18

**Conditional Probability**

a simple example:



why it's so important?

In Machine Learning, e.g. MLE

### Slide 19

**Conditional Probability**

a simple example:



why it's so important?

In Machine Learning, e.g. MLE

### Slide 20

**Conditional Probability**

a simple example:



why it's so important?

In Machine Learning, e.g. MLE

### Slide 21

**Conditional Probability**

a simple example:



why it's so important?

In Machine Learning, e.g. MLE

### Slide 22

**Conditional Probability**

a simple example:



why it's so important?

In Machine Learning, e.g. MLE

### Slide 23

**Conditional Probability**

a simple example:



why it's so important?

In Machine Learning, e.g. MLE

### Slide 24

**Conditional Probability**

a simple example:



why it's so important?

In Machine Learning, e.g. MLE

