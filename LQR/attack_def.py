import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import cvxpy as cvx
import scipy as sci
# import mosek

mosek_params={'MSK_DPAR_ANA_SOL_INFEAS_TOL':1e-6, 
              'MSK_DPAR_INTPNT_CO_TOL_PFEAS':1e-6,
              'MSK_IPAR_INFEAS_REPORT_AUTO':1}

def GetUniformAction():
    while (1):
        x = np.random.uniform() * 2.0 - 1.0;
        y = np.random.uniform() * 2.0 - 1.0;
        d = x*x+y*y 
        if d<1:
            break
    return np.array([x,y])

class dynamic(object):
    """docstring for LDS"""
    def __init__(self, A, B, Q, R, q, c, sigma):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.q = q
        self.c = c
        self.n = len(A)
        self.d = len(Q) 
        self.sigma = sigma

    def transit(self, s, a):
        r = -(s.dot(self.Q).dot(s)/2.0+self.q.dot(s)+a.dot(self.R).dot(a)+self.c)
        mean = [0 for i in range(self.n)]
        cov = np.eye(self.n)
        sp = self.A.dot(s)+self.B.dot(a)+np.random.multivariate_normal(mean, cov)*self.sigma
        return r, sp

class eqLQR(object):
    """docstring for LQR"""
    def __init__(self, gamma, eps):
        self.Ahat = None
        self.Bhat = None
        self.Qhat = None
        self.Rhat = None
        self.qhat = None
        self.chat = None
        self.K = None
        self.k = None
        self.P = None
        self.p = None
        self.gamma = gamma
        self.eps = eps

    def estimate(self, D):
        self.estimateAB(D)
        self.estimateQRqc(D)

    def setParameter(self, A, B, Q, R, q, c):
        self.Ahat = A
        self.Bhat = B
        self.Qhat = Q
        self.Rhat = R
        self.qhat = q
        self.chat = c

    def estimateAB(self, D):
        T = len(D)
        n = len(D[0][0])
        d = len(D[0][1])
        A = cvx.Variable((n, n))
        B = cvx.Variable((n, d))
        obj_MLE = 0
        for t in range(T):
            st = D[t][0]
            at = D[t][1]
            spt = D[t][3]
            obj_MLE = obj_MLE + cvx.sum_squares(A*st+B*at-spt)
        objective = cvx.Minimize(obj_MLE)
        prob = cvx.Problem(objective)
        result = prob.solve()
        self.Ahat = A.value
        self.Bhat = B.value

    def estimateQRqc(self, D):
        T = len(D)
        n = len(D[0][0])
        d = len(D[0][1])
        Q = cvx.Variable((n, n), PSD=True)
        R = cvx.Variable((d, d), PSD=True)
        q = cvx.Variable(n)
        c = cvx.Variable(1)
        obj_MSSE = 0
        for t in range(T):
            st = D[t][0]
            at = D[t][1]
            rt = D[t][2]
            exp = st.T*Q*st/2.0+q.T*st+at.T*R*at+c+rt
            obj_MSSE = obj_MSSE + cvx.sum_squares(exp)
        constraints = []
        constraints.append((R>>self.eps*np.eye(d)))
        objective = cvx.Minimize(obj_MSSE)
        prob = cvx.Problem(objective, constraints)

        # two solver options: using CVXOPT or mosek
        result = prob.solve(solver=cvx.CVXOPT)
        # result = prob.solve(solver=cvx.MOSEK, verbose=False, mosek_params=mosek_params)

        self.Qhat = Q.value
        self.Rhat = R.value
        self.qhat = q.value
        self.chat = c.value

    def solve_LQR(self):
        gamma =self.gamma
        A = self.Ahat
        B = self.Bhat
        Q = self.Qhat
        R = self.Rhat
        q = self.qhat
        P = sci.linalg.solve_discrete_are(np.sqrt(gamma)*A, np.sqrt(gamma)*B, Q, R)
        K = -gamma*np.linalg.inv(R+gamma*B.T.dot(P).dot(B)).dot(B.T).dot(P).dot(A)
        n = len(A)
        p = np.linalg.inv(np.eye(n)-gamma*(A+B.dot(K)).T).dot(q)
        k = -gamma*np.linalg.inv(R+gamma*(B.T).dot(P).dot(B)).dot(B.T).dot(p)
        self.K = K
        self.P = P
        self.p = p
        self.k = k

class attacker(object):
    """docstring for attack"""
    def __init__(self, alpha):
        self.alpha = alpha

    def attack(self, D0, learner, K, k):
        gamma = learner.gamma
        A = np.matrix(learner.Ahat)
        B = np.matrix(learner.Bhat)
        T = len(D0)
        n = len(D0[0][0])
        d = len(D0[0][1])
        r0 = [D0[t][2] for t in range(T)]
        eps = learner.eps

        r = cvx.Variable(T, name='r')
        Q = cvx.Variable((n,n), PSD=True, name='Q')
        R = cvx.Variable((d,d), PSD=True, name='R')
        p = cvx.Variable(n, name='p')
        q = cvx.Variable(n, name='q')
        c = cvx.Variable(1, name='c')
        P = cvx.Variable((n,n), PSD=True, name='P')
        attack_cost = cvx.Minimize(cvx.norm(r-r0, self.alpha))
        
        constraints = []

        # positive definite constraints with eps margin
        constraints.append((R>>eps*np.eye(d)))

        # target policy constraint
        cons1 = -gamma*B.T*P*A-(R+gamma*B.T*P*B)*K
        constraints.append((cons1==0))

        cons2 = -gamma*B.T*p-(R+gamma*B.T*P*B)*k
        constraints.append((cons2==0))

        cons3 = P-(gamma*A.T*P*(A+B*K)+Q)
        constraints.append((cons3==0))
        
        cons4 = p-(q+gamma*(A+B*K).T*p)
        constraints.append((cons4==0))

        # KKT conditions
        exp1 = 0
        exp2 = 0
        exp3 = 0
        exp4 = 0
        for t in range(T):
            st = D0[t][0]
            at = D0[t][1]
            M = np.outer(st,st)
            N = np.outer(at,at)
            temp = cvx.quad_form(st, Q)/2+cvx.quad_form(at, R)+q.T*st+c+r[t]
            exp1 = exp1+temp*M
            exp2 = exp2+temp*N
            exp3 = exp3+temp*st
            exp4 = exp4+temp
        constraints.append((exp1==0))
        constraints.append((exp2==0))
        constraints.append((exp3==0))
        constraints.append((exp4==0))

        prob = cvx.Problem(attack_cost, constraints)

        # two solver options: using CVXOPT or mosek
        prob.solve(solver=cvx.CVXOPT)
        # prob.solve(solver=cvx.MOSEK, verbose=False, mosek_params=mosek_params)

        rs = r.value
        obj = prob.value
        D = []
        for t in range(T):
            D.append((D0[t][0],D0[t][1],rs[t],D0[t][3]))

        return D, obj