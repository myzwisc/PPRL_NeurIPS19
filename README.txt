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

Important: the code for LQR by default uses solver CVXOPT. You could also use other solvers such as MOSEK, which you may need to install. To use MOSEK, uncomment line 6, 109, 199, and comment line 98, 198 in the file /LQR/attack_def.py.

For both LQR and TCE victims, if you want to reproduce the sparse attack, just find the parameter alpha in the above driver files, change its value to 1, and rerun the experiments.

Primary contact:
Yuzhe Ma (ma234@wisc.edu)