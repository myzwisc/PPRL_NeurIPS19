import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import cvxpy as cvx
import attack_def as ad

gamma = 0.9
learner = ad.TCE(gamma)

# create clean data D0, S: state space, A: action space
S = [0, 1]
A = [0, 1]
D0 = [(0,0,1,0),(0,1,0,1),(1,0,1,1),(1,1,0,0)]
r0 = np.array([D0[t][2] for t in range(len(D0))])

# learn on clean data D0
learner.estimate(D0, S, A)
Q0,Q0_traj = learner.value_iteration()
phi = learner.V
original_policy = np.argmax(Q0,axis=1)
print('original policy:')
print(original_policy)

# reward shaping
Ds = []
for (s,a,r,s_prime) in D0:
        r_shaped = r + gamma*phi[s_prime]-phi[s]
        Ds.append((s, a, r_shaped, s_prime))

# learn on shaped data
learner.estimate(Ds, S, A)
Qs,Qs_traj = learner.value_iteration()
shaped_policy = np.argmax(Qs, axis=1)
print('policy after reward shaping:')
print(shaped_policy)


# run attack
epsilon = 1
p = 2
tar_policy = [1, 1]
attacker = ad.attacker(epsilon, p)
Dp,_,obj = attacker.attack(D0, S, A, tar_policy, learner)
print('attack cost:')
print(obj)
r = np.array([Dp[t][2] for t in range(len(Dp))])

# learn on poisoned data D
learner.estimate(Dp, S, A)
Qp,Qp_traj = learner.value_iteration()

# print poisoned policy
poisoned_policy = np.argmax(Qp, axis=1)
print('posioned policy:')
print(poisoned_policy)

# check if attack is successful
poisoned_policy = np.argmax(Qp, axis=1)
for s in range(len(S)):
        if poisoned_policy[s]!=tar_policy[s]:
                print('attack failed!')
                exit()
print('attack successful!')

# plot trajectory
T0 = len(Q0_traj)
traj0_x = [Q0_traj[t][0][0] for t in range(T0)]
traj0_y = [Q0_traj[t][0][1] for t in range(T0)]

Tp = len(Qp_traj)
trajp_x = [Qp_traj[t][0][0] for t in range(Tp)]
trajp_y = [Qp_traj[t][0][1] for t in range(Tp)]

Ts = len(Qs_traj)
trajs_x = [Qs_traj[t][0][0] for t in range(Ts)]
trajs_y = [Qs_traj[t][0][1] for t in range(Ts)]

x = np.arange(-2, 12, 0.01)
y = x + 1
fig, ax = plt.subplots(1, 1)
ax.fill_between(x, y, 12, facecolor='gray', alpha=0.3)
ax.tick_params(axis='both', labelsize=15)
plt.xlabel('Q(A,stay)', fontsize=15)
plt.ylabel('Q(A,move)', fontsize=15)
plt.plot(x, x, 'k--', linewidth=1.5)
plt.plot(traj0_x, traj0_y, 'dodgerblue', marker='.', markersize=8, linewidth=3, alpha=0.4, label='clean data')
plt.plot(traj0_x[T0-1], traj0_y[T0-1], 'dodgerblue', marker='*', markersize=12, alpha=0.7)
plt.plot(trajp_x, trajp_y, 'r', marker='.', linewidth=3, markersize=8, alpha=0.4, label='poisoned data')
plt.plot(trajp_x[Tp-1], trajp_y[Tp-1], 'r', marker='*', markersize=12, alpha=0.7)
plt.plot(trajs_x, trajs_y, 'darkorange', marker='.', linewidth=3, markersize=8, alpha=0.4, label='reward shaping')
plt.plot(trajs_x[Ts-1], trajs_y[Ts-1], 'darkorange', marker='*', markersize=12, alpha=0.7)
plt.xlim([-2,12])
plt.ylim([-2,12])
plt.legend(fontsize=15)
fig.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
