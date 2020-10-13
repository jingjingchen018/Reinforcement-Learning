<<<<<<< HEAD
# Papers I read in 2020-06-06
 a recording about what I read

=======
This week I read two articles about RL and heuristic. 

[1]Samma H, Lim C P, Saleh J M. A new reinforcement learning-based memetic particle swarm optimizer[J]. Applied Soft Computing, 2016, 43: 276-297.

## 2.1 major contributions

This article is about RL and PSO. The main idea of this article is to treat the particle as an agent and then making the whole population from exploration operation switch to the convergence operation. The state space consists of five states: exploration, low-jump, high jump, fine-tuning and convergence. Setting all particles in the exploration state at first, then updating all the particles' state according the Q-value to choose the next state.

## comments

I read it roughly.

[2]Yolcu E, Poczos B. Learning Local Search Heuristics for Boolean Satisfiability[C]//Advances in Neural Information Processing Systems. 2019: 7990-8001.

## 2.2 major contributions

This paper is about using Reinforcement learning to replace the local search procedure in the framework of WalkSAT, which was used solving Boolean Satisfiability. 

The state is a pair of the problem and the solution $s=(f,X)$. 

The action function maps states to available actions, that means choose a variable to flip(0 to 1, or 1 to 0).

The transition function is to map the state-action pair to the next state. 

The reward function is 1 if the assignment(solution) $X$ is satisfying, if not, 0. 

They treat local search for SAT as an MDP and train the policy by GNN, which returns an action.


## comments
This article is novel in modeling the SAT as a graph and using GNN to train a policy. It can find satisfying assignments in fewer steps compared to a generic heuristic, but costlier.


## Research work I have done

For my personal health problem, I read some articles only.

### what issues I tried to solve: 

### progress:

### why I  succeed or why I failed. 
>>>>>>> 331ed3429029282f1062a76c69db6e2580a98092
