# POLICY ITERATION ALGORITHM

## AIM
The goal of the notebook is to implement and evaluate a policy iteration algorithm within a custom environment (gym-walk) to find the optimal policy that maximizes the agent's performance in terms of reaching a goal state with the highest probability and reward.

## PROBLEM STATEMENT
The task is to develop and apply a policy iteration algorithm to solve a grid-based environment (gym-walk). The environment consists of states the agent must navigate through to reach a goal. The agent has to learn the best sequence of actions (policy) that maximizes its chances of reaching the goal state while obtaining the highest cumulative reward.

## POLICY ITERATION ALGORITHM

Initialize: Start with a random policy for each state and initialize the value function arbitrarily.

Policy Evaluation: For each state, evaluate the current policy by computing the expected value function under the current policy.

Policy Improvement: Improve the policy by making it greedy with respect to the current value function (i.e., choose the action that maximizes the value function for each state).

Check Convergence: Repeat the evaluation and improvement steps until the policy stabilizes (i.e., when no further changes to the policy occur).

Optimal Policy: Once convergence is achieved, the policy is considered optimal, providing the best actions for the agent in each state.

## POLICY IMPROVEMENT FUNCTION
### Name: Dharini PV
### Register Number: 212222240024
```python
def policy_improvement(V,P,gamma=1.0):
  Q=np.zeros((len(P),len(P[0])),dtype=np.float64)
  for s in range(len(P)):
    for a in range(len(P[s])):
      for prob,next_state,reward,done in P[s][a]:
        Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
  new_pi=lambda s: {s:a for s, a in enumerate(np.argmax(Q,axis=1))}[s]
  return new_pi
```

## POLICY ITERATION FUNCTION
### Name: Dharini PV
### Register Number: 212222240024
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
  random_actions = np.random.choice(tuple(P[0].keys()), len(P))
  ramdon_actions=np.random.choice(tuple (P[0].keys()), len(P))
  pi=lambda s: {s: a for s, a in enumerate(random_actions)} [s]
  while True:
    old_pi={s:pi(s) for s in range (len(P))}
    V=policy_evaluation(pi,P,gamma,theta)
    pi=policy_improvement(V,P,gamma)
    if old_pi=={s:pi(s) for s in range(len(P))}:
      break
  return V,pi
```
## OUTPUT:
### Optimal policy

![image](https://github.com/user-attachments/assets/91b7888a-912b-48fc-9bd8-52905c73ae78)

### Optimal value function

![image](https://github.com/user-attachments/assets/4bb64006-1441-4ce7-9281-384a4475ad1a)

### Success rate for the Optimal policy

![image](https://github.com/user-attachments/assets/f6e8b0d1-3506-4287-ab2a-321fade98721)

## RESULT:
Thus the program to iterate the policy evaluation and policy improvement is executed successfully.
