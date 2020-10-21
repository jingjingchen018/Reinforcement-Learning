###Some interesting deductions

##  (1) What‘s the advantage of Q-value? 



Value Iteration
$$
V^{*}(s)=max_{\pi}[r(s,a)+\gamma\sum_{t=0}^{\infty}P(s^{\prime}|s,a)V^{\pi}(s^{\prime})] 
  \\ \quad = max_{\pi}\mathrm{E}[r.v.]
$$
Q-Value Iteration
$$
Q^{*}(s,a)=r(s,a)+\gamma\sum_{t=0}^{\infty}P(s^{\prime}|s,a)max_{a^{\prime}}Q^{\pi}(s^{\prime},a^{\prime})
  \\ \quad = \mathrm{E}[max(r.v.)]
$$
Then we can use **Sampling methods** to update the Q-value.



for example:

#### Sarsa (on-policy):

Sarsa and Q-learning are all iterative approximation algorithms. Sarsa starts with an arbitrary Q-function, observes transitions $(s_{k},a_{k},r_{k+1},s_{k+1},a_{k+1})$, and after transition update the Q-function with current estimate $Q_{k}(s_{k},a_{k})$ of the optimal Q-value of $(s_{k},a_{k})$ and the updated estimate $r_{k+1}+\gamma Q_{k}(s_{k+1},a_{k+1})$ in Sarsa:
$$
Q_{k+1}(s_{k},a_{k})=Q_{k}(s_{k},a_{k})+\alpha[r_{k+1}+\gamma Q_{k}(s_{k+1},a_{k+1})-Q_{k}(s_{k},a_{k})]
$$
![img](https://img-blog.csdn.net/20170519163048350?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGFuZ2xpbnpodW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

#### Q-learning (off-policy): 

Q-learning starts with an arbitrary Q-function, observes transitions $(s_{k},a_{k},r_{k+1},s_{k+1})$, and after transition update the Q-function with current estimate $Q_{k}(s_{k},a_{k})$ of the optimal Q-value of $(s_{k},a_{k})$ and the updated estimate $r_{k+1}+\gamma \max_{a^{\prime}}Q_{k}(s_{k+1},a^{\prime})$ in Q-learning:
$$
Q_{k+1}(s_{k},a_{k})=Q_{k}(s_{k},a_{k})+ \alpha_{k}[r_{k+1}+\gamma \max_{a^{\prime}}Q_{k}(s_{k+1},a^{\prime})-Q_{k}(s_{k},a_{k})]
$$
![img](https://img-blog.csdn.net/20170519163105163?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGFuZ2xpbnpodW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

Reference: (the comparison between Sarsa)

> https://blog.csdn.net/panglinzhuo/article/details/72518045

## (2) From RL to MaxEntropy RL (to RL with MaxEntropy)

### Policy Gradient Method



(2.1) e.g. from REINFORCE to PPO and TRPO



The objective of REINFORCE:


$$
J(\theta):=
$$


The objective of TRPO: 

 ## (3)  Inequalities



