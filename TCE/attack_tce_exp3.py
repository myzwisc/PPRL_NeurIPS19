import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import cvxpy as cvx
import attack_def as ad

# load grid world task
grid_task = ad.tasks()
S, A, D0, tar_policy = grid_task.load_task('exp3')
Ns = len(S)
Na = len(A)
T = len(D0)
r0 = np.array([D0[t][2] for t in range(len(D0))])

# learn on clean data D0
gamma = 0.9
learner = ad.TCE(gamma)
learner.estimate(D0, S, A)
R0 = learner.Rhat

Q0,Q0_traj = learner.value_iteration()
phi = learner.V

# print original policy
original_policy = np.argmax(Q0,axis=1)
print('Original policy:')
print(original_policy)

# run attack
epsilon = 0.1
alpha = 2.0
attacker = ad.attacker(epsilon, alpha)
Dp,_,obj = attacker.attack(D0, S, A, tar_policy, learner)
print('attack cost:')
print(obj)
r = np.array([Dp[t][2] for t in range(len(Dp))])

# learn on poisoned data D
learner.estimate(Dp, S, A)
Rp = learner.Rhat
Qp, Qp_traj = learner.value_iteration()

# print poisoned policy
poisoned_policy = np.argmax(Qp, axis=1)
print('Posioned policy:')
print(poisoned_policy)

# check if attack is successful
for s in range(len(S)):
    if poisoned_policy[s]!=tar_policy[s]:
        print('attack failed!')
        exit()
print('attack successful!')

print('||r-r_0||:')
print(np.linalg.norm(r-r0, alpha))

print('||r_0||:')
print(np.linalg.norm(r0, alpha))