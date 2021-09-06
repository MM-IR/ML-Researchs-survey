# 1.网上说ReLU的优点
1.深度网络中,能够缓解梯度消失的 情况，并且由于梯度求解非常容易等优点，就是被广泛使用。

但是，神经网络的激活函数必须是非线性的，否则，无论怎么加深网络，也仅仅只是线性函数的组合而已，仍然是线性，这样根本提升不了网络的表达能力。

**谁告诉你这个ReLU这个东西是分段函数，那么他就一定是非线性了？？？？？**

# TLDR:分析非线性其实得分析这个stack和add操作

# 2.ReLU函数的分析
首先就是对一个两层hidden layers做一个解释。

<img width="679" alt="image" src="https://user-images.githubusercontent.com/40928887/132173950-31f9584b-72b5-482b-97df-7385cff58b3b.png">

**这个就是如何我们的隐藏层输出都是ReLU函数，那么在一次前向传播和反向传播过程中，他就像一个比较细长的，表达能力较弱的网络。**

## 2.1 stack
<img width="607" alt="image" src="https://user-images.githubusercontent.com/40928887/132174569-745e3c8d-d8b7-4c0a-919e-5c94e1f810b7.png">

这个就是直接叠加操作

<img width="690" alt="image" src="https://user-images.githubusercontent.com/40928887/132174585-3f2d2a21-6c26-4f0e-b31e-022d37dc35b4.png">

**可以看出来这个叠加的结果还是一个线性函数+一个全0，所以没见到什么非线性**

## 2.2 add（非线性核心）
<img width="641" alt="image" src="https://user-images.githubusercontent.com/40928887/132174667-6afbf00a-bc68-4e3a-8ceb-ba275a9f7d5f.png">

这个就是有一点像二次函数了

## 2.3 理论上结合bias+stack+add
我们的神经网络在一层中进行下一层的propogate的时候，其实就是结合了bias（为了有negative），结合stack，结合add。

因此就是可以做到非线性。

**当然啦，这个理论上是可以拟合绝大部分在闭区间上的连续函数的。当然没办法真正意义上拟合所有函数，但是对于神经网络来说，肯定就是已经够用了。**

# 3.所以本质上这个relu就是一个多分段的线性函数
>这个除了stack+add+bias相互交融，使得ReLU有了这个非线性表达能力。

>本质上这个NN基本上都是按照mini-batch的方式进行训练，这个就是**针对于多次step来说，因为每个step训练的都是不同的细小的网络，某种程度上，也可以看成add，更加增强了非线性。有了类似ensemble的效果。**

<img width="728" alt="image" src="https://user-images.githubusercontent.com/40928887/132175380-d002f8f3-74a9-434e-a837-aa08850960cd.png">

**或许不严谨地说这个思想和ensemble如出一辙。**

