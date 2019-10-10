import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import cvxpy as cvx

class tasks(object):
    """docstring for task"""
    def __init__(self):
        self.S = {}
        self.A = {}
        self.D0 = {}
        self.tar_policy = {}
        self.tasknames = ['exp2', 'exp3']
        self.D0['exp2'] = [(0,0,-1,0),(0,1,-1,0),(0,2,-1,0),(0,3,-1,1),
          (1,0,-1,2),(1,1,-1,1),(1,2,-1,0),(1,3,-1,1),
          (2,0,-10,4),(2,1,-1,1),(2,2,-1,2),(2,3,-1,3),
          (3,0,-1,3),(3,1,-1,3),(3,2,-1,2),(3,3,-1,5),
          (4,0,-10,6),(4,1,-1,2),(4,2,-1,4),(4,3,-1,4),
          (5,0,-1,5),(5,1,-1,5),(5,2,-1,3),(5,3,-1,7),
          (6,0,-1,8),(6,1,-10,4),(6,2,-1,6),(6,3,-1,6),
          (7,0,-1,7),(7,1,-1,7),(7,2,-1,5),(7,3,-1,9),
          (8,0,2,11),(8,1,-10,6),(8,2,-1,8),(8,3,-1,10),
          (9,0,-1,12),(9,1,-1,9),(9,2,-1,7),(9,3,-1,9),
          (10,0,-1,10),(10,1,-1,10),(10,2,-1,8),(10,3,-1,13),
          (11,0,0,11),(11,1,0,11),(11,2,0,11),(11,3,0,11),
          (12,0,-1,14),(12,1,-1,9),(12,2,-1,12),(12,3,-1,12),
          (13,0,-1,13),(13,1,-1,13),(13,2,-1,10),(13,3,-1,15),
          (14,0,-1,16),(14,1,-1,12),(14,2,-1,14),(14,3,-1,14),
          (15,0,-1,17),(15,1,-1,15),(15,2,-1,13),(15,3,-1,16),
          (16,0,-1,16),(16,1,-1,14),(16,2,-1,15),(16,3,-1,16),
          (17,0,-1,17),(17,1,-1,15),(17,2,-1,17),(17,3,-1,17)]
        self.S['exp2'] = [i for i in range(18)]
        self.A['exp2'] = [0, 1, 2, 3]  #UP DOWN LEFT RIGHT
        self.tar_policy['exp2'] = [3, 0, 0, 3, 0, 3, 0, 3, 0, 0, 2, 0, 0, 2, 0, 2, 2, 1]

        self.D0['exp3'] = [(0,0,-1,2),(0,1,-1,0),(0,2,-1,0),(0,3,-1,1),
          (1,0,-1,4),(1,1,-1,1),(1,2,-1,0),(1,3,-1,3),
          (2,0,-1,5),(2,1,-1,0),(2,2,-1,2),(2,3,-1,4),
          (3,0,-1,7),(3,1,-1,3),(3,2,-1,1),(3,3,-1,6),
          (4,0,-1,8),(4,1,-1,1),(4,2,-1,2),(4,3,-1,7),
          (5,0,-1,9),(5,1,-1,2),(5,2,-1,5),(5,3,-1,8),
          (6,0,-1,11),(6,1,-1,6),(6,2,-1,3),(6,3,-1,10),
          (7,0,-1,12),(7,1,-1,3),(7,2,-1,4),(7,3,-1,11),
          (8,0,-1,13),(8,1,-1,4),(8,2,-1,5),(8,3,-1,12),
          (9,0,-1,14),(9,1,-1,5),(9,2,-1,9),(9,3,-1,13),
          (10,0,-1,16),(10,1,-1,10),(10,2,-1,6),(10,3,-1,15),
          (11,0,-1,17),(11,1,-1,6),(11,2,-1,7),(11,3,-1,16),
          (12,0,-1,18),(12,1,-1,7),(12,2,-1,8),(12,3,-1,17),
          (13,0,-1,13),(13,1,-1,8),(13,2,-1,9),(13,3,-1,18),
          (14,0,1,19),(14,1,-1,9),(14,2,-1,14),(14,3,-1,14),
          (15,0,-1,20),(15,1,-1,15),(15,2,-1,10),(15,3,-1,15),
          (16,0,-1,21),(16,1,-1,10),(16,2,-1,11),(16,3,-1,20),
          (17,0,-1,22),(17,1,-1,11),(17,2,-1,12),(17,3,-1,21),
          (18,0,-1,23),(18,1,-1,12),(18,2,-1,13),(18,3,-1,22),
          (19,0,0,19),(19,1,0,19),(19,2,0,19),(19,3,0,19),
          (20,0,-1,24),(20,1,-1,15),(20,2,-1,16),(20,3,-1,20),
          (21,0,-1,25),(21,1,-1,16),(21,2,-1,17),(21,3,-1,24),
          (22,0,-1,26),(22,1,-1,17),(22,2,-1,18),(22,3,-1,25),
          (23,0,-1,27),(23,1,-1,18),(23,2,-1,23),(23,3,-1,26),
          (24,0,-1,28),(24,1,-1,20),(24,2,-1,21),(24,3,-1,24),
          (25,0,-1,25),(25,1,-1,21),(25,2,-1,22),(25,3,-1,28),
          (26,0,-1,29),(26,1,-1,22),(26,2,-1,23),(26,3,-1,26),
          (27,0,-1,27),(27,1,-1,23),(27,2,-1,27),(27,3,-1,29),
          (28,0,-1,30),(28,1,-1,24),(28,2,-1,25),(28,3,-1,28),
          (29,0,-1,29),(29,1,-1,26),(29,2,-1,27),(29,3,-1,29),
          (30,0,2,31),(30,1,-1,28),(30,2,-1,30),(30,3,-1,30),
          (31,0,0,31),(31,1,0,31),(31,2,0,31),(31,3,0,31)]
        self.S['exp3'] = [i for i in range(32)]
        self.A['exp3'] = [0, 1, 2, 3] #UP DOWN LEFT RIGHT
        self.tar_policy['exp3'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
        0, 0, 0, 0, 3, 0, 0, 0, 3, 1, 0, 3, 1, 1, 0, 1, 0, 0]

    def load_task(self, task_name):
        if task_name not in self.tasknames:
             raise NameError('Error: task must be exp2 or exp3')
        else:
             return self.S[task_name], self.A[task_name], self.D0[task_name], self.tar_policy[task_name]

class MDP(object):
    """docstring for MDP"""
    def __init__(self, S, A, R, P, sigma):
        self.S = S
        self.A = A
        self.R = R
        self.P = P
        self.sigma = sigma

    def transit(self, s, a):
        r = np.random.normal(self.R(s,a),sigma)
        sp = list(numpy.random.multinomial(1, P(s,a))).index(1)
        return r, self.S[sp]

class TCE(object):
    """docstring for TCE"""
    def __init__(self, gamma):
        self.S = None
        self.A = None
        self.Ns = None
        self.Na = None
        self.Rhat = None
        self.Phat = None
        self.gamma = gamma
        self.Q = None
        self.V = None

    def estimate(self, D, S, A):
        self.S = S
        self.Ns = len(S)
        self.A = A
        self.Na = len(A)
        self.Rhat = np.array([[0.0 for j in range(self.Na)] for i in range(self.Ns)])
        self.Phat = {}
        for s in S:
            for a in A:
                self.Phat[(s,a)]=np.array([0.0 for i in range(self.Ns)])

        T = len(D)
        for s in S:
            for a in A:
                Tsa = [t for t in range(T) if D[t][0]==s and D[t][1]==a]
                r_sa = [D[t][2] for t in Tsa]
                sp_sa = [D[t][3] for t in Tsa]
                Nsa = len(Tsa)
                self.Rhat[s,a] = np.sum(r_sa)/len(r_sa)
                for sp in S:
                    self.Phat[(s,a)][sp] = float(sp_sa.count(sp))/Nsa

    def value_iteration(self):
        tol = 1e-30
        Q = np.array([[0.0 for j in range(self.Na)] for i in range(self.Ns)])
        Q_traj={}
        t = 0
        Q_traj[t] = cp.deepcopy(Q)
        while True:
            t = t + 1
            Q_pre = cp.deepcopy(Q)
            for s in range(self.Ns):
                for a in range(self.Na):
                    Q[s,a] = self.Rhat[s,a]+self.gamma*self.Phat[s,a].dot(np.max(Q_pre,axis=1))
            Q_traj[t] = cp.deepcopy(Q)
            if np.max(np.abs(Q-Q_pre))<tol:
                break
        self.Q = Q
        self.V = np.max(Q,axis=1)
        return Q, Q_traj

class attacker(object):
    """docstring for attack"""
    def __init__(self, epsilon, p):
        self.epsilon = epsilon
        self.p = p

    def attack(self, D0, S, A, tar_policy, learner):
        Ns = len(S)
        Na = len(A)
        T = len(D0)
        r0 = [D0[t][2] for t in range(T)]

        r = cvx.Variable(T)
        Q = cvx.Variable((Ns, Na))
        attack_cost = cvx.Minimize(cvx.norm(r-r0, self.p))
        constraints = []
        for s in S:
            for a in A:
                if a != tar_policy[s]:
                    # target policy constraint
                    cons = (Q[s,tar_policy[s]]>=Q[s,a]+self.epsilon)
                    constraints.append(cons)

        # Bellman optimality equation constraint
        Rhat = learner.Rhat
        Phat = learner.Phat
        gamma = learner.gamma
        Qmax = [Q[s,tar_policy[s]] for s in range(Ns)]
        for s in S:
            for a in A:
                Tsa = [t for t in range(T) if D0[t][0]==s and D0[t][1]==a]
                cons = (Q[s,a] == np.mean([r[t] for t in Tsa])+gamma*Phat[(s,a)].dot(Qmax))
                constraints.append(cons)
        prob = cvx.Problem(attack_cost, constraints)
        prob.solve(solver='ECOS', max_iters=5000, reltol=1e-20, abstol=1e-20)
        Q = Q.value
        r = r.value
        obj = prob.value
        D = []
        for t in range(T):
            D.append((D0[t][0],D0[t][1],r[t],D0[t][3]))
        return D, Q, obj