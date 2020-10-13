## MARL_survey

### *P1 Multi-agent reinforcement learning: An overview*

```
@incollection{bucsoniu2010multi,
  title={Multi-agent reinforcement learning: An overview},
  author={Bu{\c{s}}oniu, Lucian and Babu{\v{s}}ka, Robert and De Schutter, Bart},
  booktitle={Innovations in multi-agent systems and applications-1},
  pages={183--221},
  year={2010},
  publisher={Springer}
}
```

### *P2 A Comprehensive Survey of Multiagent Reinforcement Learning*

```
@article{busoniu2008comprehensive,
  title={A comprehensive survey of multiagent reinforcement learning},
  author={Busoniu, Lucian and Babuska, Robert and De Schutter, Bart},
  journal={IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews)},
  volume={38},
  number={2},
  pages={156--172},
  year={2008},
  publisher={IEEE}
}
```

### Definition in Math



## Tasks types in MARL 

There are three tasks in MARL 

Fully cooperative, fully competitive, and more general (neither cooperative or competitive).



### Representative algorithms 

![image-20201013171013412](/Users/chenjingjing/Library/Application Support/typora-user-images/image-20201013171013412.png)

### Advantages 

Compared to single agent RL



### Challenges

*caused by RL* 

(a)  curse of dimensionality  (states space)

(b)  exploration and exploitation trade-off (methods: $\epsilon$-greedy, Boltzmann exploration policy, in state $s$ selects action $a$ with probability
$$
h(s, a)=\frac{e^{Q(s, a) / \tau}}{\sum_{\bar{a}} e^{Q(s, \bar{a}) / \tau}},
$$
where $\tau$>0, the temperature, controls the randomness of the exploration. When $\tau \rightarrow 0$, the equation becomes equivalent with greedy selection. When $\tau \rightarrow \infty$, action selection is purely random. For $\tau \in (0,\infty)$, higher-valued actions have a greater chance of being selected than lower-valued ones.

*caused by Multi-agents*

(a) 

(b)

(c)