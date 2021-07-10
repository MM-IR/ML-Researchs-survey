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

