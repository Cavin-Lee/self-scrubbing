# self-scrubbing
FBN estimation with self scrubbing operation
Several illustrations:
1. Run brainNetworkLearning_test.m for constructing functional brain networks based on different methods (with different parameters); then, run classifierCVforL1L2.m for selecting optimal regularized parameters (based on an inner LOO cross validation) and classification.
2. The codes of the proposed brain network construction algorithms are created based on Liu et al.'s SLEP toolbox (http://www.yelab.net/software/SLEP/) by rewriting the accel_grad_mlr.m file for dealing with l1- and trace-norm simultaneously. Classification is based on Lin et al.'s LIBSVM toolbox (https://www.csie.ntu.edu.tw/~cjlin/libsvm/). All copyright of these toolboxes reserved by the original authors.
3. Please feel free to contact Lishan Qiao (qiaolishan@nuaa.edu.cn; qiaolishan@lcu.edu.cn) if having any question.
