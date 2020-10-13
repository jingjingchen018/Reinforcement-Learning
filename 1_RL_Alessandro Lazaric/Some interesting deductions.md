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

Sara and Q-learning are all iterative approximation algorithms. They all start with an arbitrary Q-function, observes transitions $(s_{k},a_{k},r_{k+1},s_{k+1})$, and after transition update the Q-function with current estimate $Q_{k}(s_{k},a_{k})$ of the optimal Q-value of $(s_{k},a_{k})$ and the updated estimate $r_{k+1}+\gamma Q_{k}(s_{k+1},a_{k+1})$ in Sara or $r_{k+1}+\gamma \max_{a^{\prime}}Q_{k}(s_{k+1},a^{\prime})$ in Q-learning:



Sara (on-policy):
$$
Q_{k+1}(s_{k},a_{k})=Q_{k}(s_{k},a_{k})+\alpha[r_{k+1}+\gamma Q_{k}(s_{k+1},a_{k+1})-Q_{k}(s_{k},a_{k})]
$$


Q-learning (off-policy): 
$$
Q_{k+1}(s_{k},a_{k})=Q_{k}(s_{k},a_{k})+ \alpha_{k}[r_{k+1}+\gamma \max_{a^{\prime}}Q_{k}(s_{k+1},a^{\prime})-Q_{k}(s_{k},a_{k})]
$$


## (2) From RL to MaxEntropy RL (to RL with MaxEntropy)



(2.1) e.g. from REINFORCE to PPO and TRPO



The objective of REINFORCE:





The objective of TRPO:

