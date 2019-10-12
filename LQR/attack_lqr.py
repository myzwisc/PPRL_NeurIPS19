import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import cvxpy as cvx
import attack_def as ad
import scipy as sci

# set random seed
np.random.seed(1)

# time interval 0.1s
h = 0.1

# vehicle mass
m = 1

# friction parameter
eta = 0.5

# parameters of the dynamic system
A = np.array([[1, 0, h, 0],
               [0, 1, 0, h],
               [0, 0, 1-h*eta/m, 0],
               [0, 0, 0, 1-h*eta/m]])

B = np.array([[0, 0],
     [0, 0],
     [h/m, 0],
     [0, h/m]])

Q = np.eye(4)

R = np.eye(2)*0.1

q = np.array([0 for i in range(4)])

c = 0

# state transition noise
sigma = 1e-2

# define dynamic system
dyn = ad.dynamic(A,B,Q,R,q,c,sigma)

# learner parameter
gamma = 0.9
eps = 1e-2

# define LQR controller
eqLQR = ad.eqLQR(gamma, eps)
eqLQR.setParameter(A,B,Q,R,q,c)
eqLQR.solve_LQR()
K_true = eqLQR.K
k_true = eqLQR.k
print('true optimal policy: ')
print('K: ')
print(K_true)
print('k: ')
print(k_true)

# time horizon
T = 400

# generate clean data D0
s = {}
s[0] = np.array([1,1,1,-0.5])
a = {}
r = {}
D0 = {}
for t in range(T):
    a[t] = ad.GetUniformAction()
    r[t],s[t+1] = dyn.transit(s[t],a[t])
for t in range(T):
    D0[t] = (s[t],a[t],r[t],s[t+1])

# learn on clean data
eqLQR.estimate(D0)
Ahat = eqLQR.Ahat
Bhat = eqLQR.Bhat
eqLQR.solve_LQR()
print('original policy: ')
print('K: ')
print(eqLQR.K)
print('k: ')
print(eqLQR.k)

# produce clean trajectory
traj_s = {}
traj_s[0] = np.array([1,1,1,-0.5])
T_test = 3000
for t in range(T_test):
    at = eqLQR.K.dot(traj_s[t])+eqLQR.k
    _,traj_s[t+1] = dyn.transit(traj_s[t],at)
X0 = [traj_s[t][0] for t in range(T_test)]
Y0 = [traj_s[t][1] for t in range(T_test)]

# parameters of target dynamical system (used to compute target policy)
s_target = np.array([0,1,0,0])
Q_target = cp.deepcopy(Q)
R_target = cp.deepcopy(R)
q_target = -s_target
c_target = s_target.dot(Q_target).dot(s_target)/2.0

eqLQR.setParameter(Ahat,Bhat,Q_target,R_target,q_target,c_target)
eqLQR.solve_LQR()
K_dagger = eqLQR.K
k_dagger = eqLQR.k
print('target policy: ')
print('K: ')
print(K_dagger)
print('k: ')
print(k_dagger)

# p-norm
alpha = 2.0

# perform attack
eqLQR.estimateAB(D0)
attacker = ad.attacker(alpha)
Dp, obj = attacker.attack(D0, eqLQR, K_dagger, k_dagger)

# learn on poisoned data
eqLQR.estimate(Dp)
eqLQR.solve_LQR()
Kp = eqLQR.K
kp = eqLQR.k
print('poisoned policy:')
print('K: ')
print(Kp)
print('k: ')
print(kp)

# produce poisoned trajectory
traj_s = {}
traj_s[0] = np.array([1,1,1,-0.5])
for t in range(T_test):
    at = Kp.dot(traj_s[t])+kp
    _,traj_s[t+1] = dyn.transit(traj_s[t],at)
Xp = [traj_s[t][0] for t in range(T_test)]
Yp = [traj_s[t][1] for t in range(T_test)]

# plot clean and poisoned trajectory
plt.plot(X0,Y0,'-', marker='.',label='clean data')
plt.plot(Xp,Yp,'-', marker='.',label='poisoned data')
plt.plot(s_target[0],s_target[1],'+',markersize=15,markeredgewidth=3,label='target')
plt.legend(fontsize=15)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.tick_params(labelsize=15)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# plot clean and poisoned rewards
r0 = np.array([D0[t][2] for t in range(T)])
r = np.array([Dp[t][2] for t in range(T)])
print('||r-r_0||:')
print(np.linalg.norm(r-r0, alpha))
print('||r_0||:')
print(np.linalg.norm(r0, alpha))

plt.figure()
plt.subplot(211)
plt.plot(r0,'o', markersize=3, label=r'clean rewards $r^0$')
plt.plot(r,'o', markersize=3, label=r'poisoned rewards $r$')
plt.legend(fontsize=15,markerscale=2)
plt.tick_params(labelsize=15)
plt.xlabel(r'$t$', fontsize=15)

plt.subplot(212)
plt.plot(r-r0,'o', markersize=3, label=r'$r-r^0$',color='#2ca02c')
plt.legend(fontsize=15,markerscale=2)
plt.tick_params(labelsize=15)
plt.xlabel(r'$t$', fontsize=15)
plt.show()