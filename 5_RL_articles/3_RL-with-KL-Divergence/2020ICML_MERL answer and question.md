# articles about entropy and RL

## （2020ICLR）If MaxEnt RL is the Answer, What is the **Question**

```
@article{eysenbach2019if,
  title={If MaxEnt RL is the Answer, What is the Question?},
  author={Eysenbach, Benjamin and Levine, Sergey},
  journal={arXiv preprint arXiv:1910.01913},
  year={2019}
}
```

出自卡耐基梅隆, UC 伯克利 google brain, 

**Abstracts**:

Experimentally, it has been observed that humans and animals often make decisions that do not maximize their expected utility, but rather choose outcomes randomly, **with probability proportional to expected utility**. **Probability matching**, as this strategy is called, is equivalent to **maximum entropy reinforcement learning (MaxEnt RL)**. However, MaxEnt RL does not optimize expected utility. In this paper, we formally show that MaxEnt RL does optimally solve certain classes of control problems with **variability 可变性，变化性 in the reward function**. In particular, we show (1) that MaxEnt RL can be used to solve **a certain class of POMDPs**, and (2) that MaxEnt RL is equivalent to a two-player game where an adversary **对手**chooses the reward function. These results suggest a **deeper connection between MaxEnt RL, robust control**, and **POMDPs**, and **provide insight for the types of problems** for which we might expect MaxEnt RL to produce effective solutions. Specifically, our results suggest that **domains with uncertainty in the task goal** may be **especially well-suited for MaxEnt RL methods**

A.3 WHEN CAN TRAJECTORY-LEVEL REWARDS BE DECOMPOSED?

a trajectory-level reward function $r_{τ} (τ )$ 

$p_{r_{\tau}}(\tau) \propto p_{1}\left(s_{1}\right) e^{r_{\tau}(\tau)} \prod_{t=1}^{T} p\left(s_{t+1} \mid s_{t}, a_{t}\right)$

B.8 ALL ADVERSARIAL GAMES ARE MAXENT PROBLEMS 

所有的对抗游戏都是最大熵问题。

 **Comments**

**Decision:** Reject

**Comment:** This paper studies maximum entropy reinforcement learning in more detail. Maximum entropy is a popular strategy in modern RL methods and also seems used in human and animal decision making. However, it does not lead to optimize expected utility. The authors propose a setting in which maximum entropy RL is an optimal solution.  

The authors were quite split分开的 on the paper, and there has been an animated discussion热烈的讨论 between the reviewers among each other and with the authors.  

The technical quality is good, although one reviewer commented on the restricted setting of the experiments (bandit problems). The authors have addressed this by adding an additional experiment. Futhermore, two reviewers commented that the clarity of the paper could be improved.  

A larger part of the discussion (also the private discussion) revolved around relevance and significance相关性和意义, especially of the meta-pomdp setting that takes up a large part of the manuscript. 尤其是什么占了很大的篇幅 - A reviewer mentioned that after reading the paper, **it does not become more clear why maximum entropy RL works well in practice**. The discussion even turned to why MaxEntropyRL might be *unreasonable* from the point of view of needing a meta-POMDP with Markov assumptions, which doesn't help shed light on its empirical success. The meta-POMDP setting does not seem to reflect the use cases where maximum entropy RL has done well in emperical studies.  Meta-POMDP的背景并没有反映出最大熵RL 在emperical studies上面的使用 - Another reviewer mentioned that **earlier papers have investigated maximum entropy RL**, and that the paper tries to offer a new perspective with the Meta-POMDP setting. The discussion of this discussion was not deemed complete in current state and needs more attention (splitting the paper into two along these lines is a possibility mooted by two of the reviewers). A particular example was the doctor-patient example, where in the meta-POMDP setting the doctor would repeatedly attempt to cure a fixed sampled illness, rather than e.g. solving for a new illness each time.  **Based on the discussion, I would conclude that the topic broached by the paper is very relevant and timely, however, that the paper would benefit from a round of major revision and resubmission rather than being accepted to ICLR in current form. ** 大修

#### (1)accept 意见模版

**Review:** The paper investigates the reason behind the success of MaxEntropy in reinforcement learning theoretically, connecting it to  robust control. 

I  think the paper should be accepted, as it investigates an important approach and offers useful insight. I think that the robust reward is an interesting perspective and the paper is also well written. 

I need to remark that I am not familiar enough with RL theory literature to know of novel this work is,. 

Detailed remarks: 

- There is an error in the proof in A.1. You are optimizing a functional not a function, so the cannot simply use Lagrange multipliers (also the derivation ignores the integral). The problem is solved easily with (constrained) Euler-Lagrange and must be corrected. 

- I disagree with the exploration paragraph in sec. 2. While the final applied agent might be deterministic, using a stochastic agent for exploration during learning is helpful. Exploration might not be only (or main) motivation behind MaxEntRL but it still a good motivation. 

- I was happy to see the limitation of lemma 4.1 stated clearly, as some paper are less honest with their limitations. 

- You write "Only the oracle version of fictitious play, which makes assumptions not made by MaxEnt", maybe I missed it but I didn't see what assumptions the oracle made.

- MaxEnt has been very successful in inverse RL where we try to find the reward, which seems connected to the conclusions here about robustness to reward perturbations. While adding analysis on IRL might be outside the scope of this paper, something at least should be said about maxEnt and IRL and the connection to the current results. 

  

  Typo: "inference problem be defining" -> "inference problem by defining"



**回复Comment:** Thanks for the review and feedback for improving the paper! 1. Typo in proof in A.1: Thanks for pointing this out this typo! We should be optimizing w.r.t. \pi(\tau), so the corresponding derivative should be dL/d\pi(\tau). We have fixed this typo and clarified that we are using constrained Euler-Langrange. 2. Exploration: Yes, we agree that MaxEnt RL empirically performs good exploration. We have added a sentence to the Exploration paragraph in Section 2 to not dismiss this motivation. 3. Oracle fictitious play: The fictitious play baseline has access to the rewards for all arms, not just the arm it selected (we clarified this halfway through the prior paragraph.). In contrast, the MaxEnt RL method only observes the rewards for the arm that it actually pulls. 4. MaxEnt IRL: Thanks for this great suggestion! IRL is an ill-posed problem, so it lends itself quite naturally to approaches that can cope with reward variability. We conjecture (but have not proven) that MaxEnt IRL yields reward functions which, when combined with MaxEnt RL, guarantee that the resulting policy perform well on the unknown reward function. We have added some discussion of this to the conclusion.

### （2）weaken reject

summary： The paper discusses the use of maximum entropy in Reinforcement Learning. **Specifically**, it relates the solution of the maximum entropy RL problem to the solutions of two different settings, 1) a ‘meta-POMDP’ regret minimization problem and 2) a ‘robust reward control’ problem. Both cases follow with simple experiments. 

 **I feel the paper could have been written more clearly**. There seem to be too many definitions and descriptive examples that diverge the attention of the reader from the main problem setting. There are **quite a bit of grammatical errors** in the paper, making it even harder to follow. With these many definitions in the text, **it is hard to make out the actual contributions of the work**.  **实验有限**Moreover, the experiments are restricted to the bandit setting and do not provide any empirical evidence on the MDP centered theory. Overall, although the paper does well in motivating the problem, the lack of rigorous experiments and **poorly structured writing** advocate for a weak rejection.  

Comments/questions: 

- Can the authors comment on why it makes intuitive sense to study the meta-POMDP and robust reward control problem settings together? I see the commonality being the reward variability, but is there something else?
- If one wants to solve the meta-POMDP through max entropy RL, how general/strong is the assumption that we are given access to the target trajectory belief?
- In the goal reaching meta-POMDP, it makes sense to only have the final state distribution in the definition. What does the action taken in the final state signify? 
- It would be more intuitive to note the optimal solution as pi* and not pi (Lemma 4.1). -
- In the meta-POMDP, does the task change after every meta-episode? 
- I think it would be better to have separate, consistently named subsections devoted to defining the two problem settings and then move on to proving equivalence with the max entropy case.

**Rating:** 3: Weak Reject

**Experience Assessment:** I have published one or two papers in this area.

**Review Assessment: Thoroughness In Paper Reading:** I read the paper thoroughly.

**Review Assessment: Checking Correctness Of Derivations And Theory:** I assessed the sensibility of the derivations and theory.

**Review Assessment: Checking Correctness Of Experiments:** I assessed the sensibility of the experiments.

