<img width="663" alt="image" src="https://user-images.githubusercontent.com/40928887/125154327-1c377680-e18c-11eb-92d2-440f9e7459e3.png">
# 1.abstract
>We characterize a prevalent weakness of deep neural networks(DNN) - overthinking.描述xxx的一个缺点。

这个就是DNN可以reach正确的predictions before最后的layer。

>overthinking: 这个是计算上的wasteful的，it can also be destructive when, by the final layer, a correct prediction changes into a misclassification.

<img width="272" alt="image" src="https://user-images.githubusercontent.com/40928887/125154787-71748780-e18e-11eb-8ab7-3273eba6a9ed.png">

## For prediction transparency,
我们这里就是提出了一个Shallow-Deep network, a generic modification to off-the-shelf DNNs that introduces internal classifiers.

**这里就是四个modern architectures，在三个任务上训练，to characterize the overthinking problem.**

a DNN’s sequence of layers resembles human perception in the way it combines simple representations, such as edges, into more complex ones, such as
faces

>这个本质上也是人有的问题，如果oberthinking甚至是destructive的。

## Motivation
就是如果人有这样的问题，可是forward的network有没有呢？

**我们针对DNN的话都是采用的same forward architecture, 我们这里问的问题就是: Are deep neural Networks also susceptible to overthinking?**
我们就是说不同samples这个layers数目确实是有不少effect的。

>与之有关的都是先前的有关early exit的工作或者说error indicators.

# 2.我们的贡献
1.Shallow-Deep Network(SDN), a generic modification to off-the-shelf DNNs for introducing *internal classifiers*

2.我们这里就是IC在各种各样的stages of the forward pass.

**我们这里就是有效combine shallower and deeper networks into one。**

这里的feature reduction in an IC acts as a regularizer, and ensures that our method can scale up to large DNNs.

<img width="430" alt="image" src="https://user-images.githubusercontent.com/40928887/125159833-7fd19c00-e1ac-11eb-8e8e-aacdba475727.png">

>我们可以把我们的这个设计用在both pre-trained and untrained DNNs.

3.conversion from a pre-trained DNN is efficient as we only train the parameters in the ICs.
>除此以外，通过使用一个weighted loss function，我们这里就是可以训练原始的network from scratch一起with the ICs.

>这里的mode of training, the SDN training, often improves the original accuracy and yields more accurate ICs.

## 第二个贡献就是
>show that CNN overthink on the majority of inputs.both *wasteful and destructive.*

1.Our experiments show that, for up to ∼95% of instances, overthinking leads to slow inferences and wasted computation.
**这个就是说明complex输入@需要full network depth的本身就是uncommon**

2.destructive effect of overthinking in up to **50%**的CNN errors，

we show that a recent
backdooring attack on CNNs (Gu et al., 2017) induces the
same effect on the victim network.

## 第三个contribution
>这个就是人类的话就是不能evaluate perfectly whether分类是正确的，但是我们可以使用heuristics against overthinking。

**这里的创新点就是confidence-based early exits and analysis of confusion.**

>1.我们的第一个heuristic就是使用confidence of an internal prediction来asess its correctness。

这个heuristic就是我们可以直接检测什么时候我们这个网络需要stop thinking以及make an early prediction@EARLY EXIT.

**我们这里就是without任何的loss of acc，我们可以减少这个avergae inference cost直到75%的，以及减轻这个wasteful effect of overthinking**

>disagreement among the internal predictions hint the state of confusion the network is in.

*这个confusion呢就是伤害了这个correct internal prediction以及导致这个输入bypass the early exits.*

>2.我们这设计的就是一个新的new confusion metric来量化这个内部的disagreement。

这个confusion scores就是可靠地indicate是否这个网络很可能去misclassify an input。

通过调查这个confusion，我们就是可以investigate the input elements that cause the confusion.

**这里以欧两个作用, In addition to their practical application regarding diagnosing DNN errors, these visualizations provide a new perspective for reasoning about model interpretability.**

## 我们这里的实验设置
>这里就是evaluate our techniques on three tasks:

1)CIFAR-10/100/Tiny ImageNet.

这里就是apply the SDN modification to 4 off-the-shelf CNN architectures: VGG/ResNet/Wide-ResNet/MobileNets.

**我们的SDN的early exits策略就是mitigate the wasteful effect of overthinking and cut the avergae inference costs by more than 50% in CIFIR-x, 25% in Tiny ImageNet.**
>我们的early exit can improve a CNN's accuracy by up to 8%, and recovers the acc of a backdoored CNN from 12 to 84%.

<img width="409" alt="image" src="https://user-images.githubusercontent.com/40928887/125160616-819d5e80-e1b0-11eb-8459-49adfa6c22bf.png">

<img width="423" alt="image" src="https://user-images.githubusercontent.com/40928887/125160639-9843b580-e1b0-11eb-948f-df361824a309.png">

## 为了quantify the test-time inference cost of a network, we measure the average number of floating point operations (FLOPs), a network performs to classify an input.
>分类performance，我们这里就是simple report the TOP-1 acc on the test data.

# 3.Shallow-Deep Network
>1.Setting: 我们这里就是考虑supervised learning setting, and the standard DNN structure: a sequence of internal layers ending with a final classifer.

## 3.1 Attaching the internal classifiers.
1)FN :1个feature scaling就是reduction@我们用mix pooling@average结合max，然后就有一个linear softmax。

<img width="413" alt="image" src="https://user-images.githubusercontent.com/40928887/125161759-b4e2ec00-e1b6-11eb-9ebd-78ec1bf86809.png">

## 3.2 我们的IC放在哪里呢
我们这里就是pick the internal layers closest to the 15%/30%/45%/60%/75%/90% of the full network's inference cost.

我们这里几乎是6个internal predictions以及一个final的。

<img width="409" alt="image" src="https://user-images.githubusercontent.com/40928887/125161827-1c00a080-e1b7-11eb-92fa-fc5fd8611775.png">

## 3.3 我们是如何训练的呢/
1.IC-only Training: 这个就是一般两种training strategy: 这个就是取决于咱们original CNN是不是pretrained
>1.IC only training.

>2.SDN training.

### 3.3.1 IC-only training
>freeze original weights, and train only the weights in the attached ICs.

>我们这里有一个major drawback这里就是标准的网络仅仅是改进最后的final accuracy,
>我们这里就是使用一个weighted loss function就是去优化所有的internal classifier。

<img width="375" alt="image" src="https://user-images.githubusercontent.com/40928887/125162432-561f7180-e1ba-11eb-9a2d-1fb5902a23f9.png">

# 5.1 Confidence-Based Early Exits
这个就是借助中间的layers的IC的confidence来决定是否还要继续forward。

如果exceeds the threshold parameter q.

>如果我们的q value设置的比较大，那么这里就是conservative的，and in turn, reduce the early exit rates.

in turn就是接着。

**我们这里就是使用一个small unlabeled holdout set去search for a q value to satisfy our computational constraints.**

## 5.1.1 Early Exits Mitigate the Wastful Effect
>






<img width="423" alt="image" src="https://user-images.githubusercontent.com/40928887/125160639-9843b580-e1b0-11eb-948f-df361824a309.png">这里
<img width="423" alt="image" src="https://user-images.githubusercontent.com/40928887/125160639-9843b580-e1b0-11eb-948f-df361824a309.png"> 
