This folder contains the code for the following paper:

Yuzhe Ma, Xuezhou Zhang, Wen Sun, Xiaojin Zhu. Policy Poisoning in Batch Reinforcement Learning and Control. In The 33rd Conference on Neural Information Processing Systems (NeurIPS), 2019.

(1). The code to perform attack of the TCE victim is in subfolder TCE/. There are three driver files:

attack_tce_exp1.py
attack_tce_exp2.py
attack_tce_exp3.py

corresponding to experiment 1, 2, and 3 in the paper respectively. The file attack_def.py contains the relevant functions used in the above three driver files. To run any of the above script, use command

python attack_tce_xxxx.py

(2). The code to perform attack of the LQR victim is in subfolder LQR/. To run attack of LQR, use command

python attack_lqr.py

Important: the code for LQR by default uses solver MOSEK, which you may need to install. However, the more commonly used CVXOPT solver can also work well. To use CVXOPT, comment line 6, 108, 198, and uncomment line 109, 199.


Primary contact:
Yuzhe Ma (ma234@wisc.edu)