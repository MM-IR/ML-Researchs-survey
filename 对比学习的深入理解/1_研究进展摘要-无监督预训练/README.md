对比学习(Contrastive Learning)最近一年比较火，各路大神比如Hinton、Yann LeCun、Kaiming He及一流研究机构比如Facebook、Google、DeepMind，都投入其中并快速提出各种改进模型：Moco系列、SimCLR系列、BYOL、SwAV…..，各种方法相互借鉴，又各有创新，俨然一场机器学习领域的军备竞赛。对比学习属于无监督或者自监督学习，但是目前多个模型的效果已超过了有监督模型，这样的结果很令人振奋。

>对比学习属于无监督或者自监督学习，但是目前多个模型的效果已经超过了**有监督学习，这样的结果很令人振奋。**

# 1.BERT成功的推进
NLP的Bert模型对于**这波图像领域的对比学习热潮**，是具有启发和推动作用的。我们知道，Bert预训练模型，通过MLM任务的自监督学习，充分挖掘了模型从海量无标注文本中学习通用知识的能力。而图像领域的预训练，往往是有监督的，就是用ImageNet来进行预训练，但是在下游任务中Fine-tuning的效果，跟Bert在NLP下游任务中带来的性能提升，是没法比的。

>NLP做自监督成功了，图像领域难道就不可以成功吗？

**这个就是让图像领域从有监督预训练到自监督预训练。**

<img width="665" alt="image" src="https://user-images.githubusercontent.com/40928887/127965507-d189cef0-8f02-410a-8b68-ade7d4fd69c5.png">

>图像想要发展也得沿用这个策略。

# 2.对比学习就是想要干NLP类似BERT的事情
对比学习是自监督学习的一种，也就是说，不依赖标注数据，要从无标注图像中自己学习知识。

>自监督学习其实在图像中已经被探索很久了，两大类比如生成式自监督学习和判别式自监督学习。

1)generative: VAE/GAN, 即它要求模型重建图像或者图像的一部分，这类型的任务难度相对比较高，要求像素级的重构，中间的图像编码必须包含很多细节信息。

2)discriminative: 对比学习，这个任务难度要低一些，**无明确定义、有指导原则**。

>它的指导原则是：通过自动构造相似实例和不相似实例，要求习得一个表示学习模型，通过这个模型，使得相似的实例在投影空间中比较接近，而不相似的实例在投影空间中距离比较远

1.如何构造相似实例;

2.如何构造不相似实例;

3.如何构造能够遵循上述指导原则的表示学习模型结构

4.如何防止模型坍塌(Model Collapse)


<img width="668" alt="image" src="https://user-images.githubusercontent.com/40928887/127965776-9b63deeb-3967-413e-a3c1-d231e99399f9.png">

# 3.我们首先介绍一个标准的模型case: 基于负例的对比学习: SimCLR为例
<img width="864" alt="image" src="https://user-images.githubusercontent.com/40928887/127968450-3050eaec-e40f-4dcb-9ac3-c61e2da320a5.png">

>首先我们SimCLR是如何构造正例和负例呢？

<img width="860" alt="image" src="https://user-images.githubusercontent.com/40928887/127968496-4f6a9da9-4f05-4480-a150-dff5f7d14e22.png">

我们就是**数据增强之后的图像相互成为positive case。**

>对比学习就是希望学习某个表示模型，它能够将图片映射到某个投影空间，并且在这个空间内拉近正例的距离，推远负例的距离。

**迫使表示模型能够忽略表面因素，学习图像的内在一致结构信息，即学会某些类型的不变性，比如遮挡不变性、旋转不变性、颜色不变性等。**

>SimCLR证明了，如果能够同时融合多种图像增强操作，增加对比学习模型任务难度，对于对比学习效果有明显提升作用。

## 3.1 双塔结构，不过这里叫two branch
<img width="858" alt="image" src="https://user-images.githubusercontent.com/40928887/127968751-4b1b5c5f-a234-465c-9c82-c6a822d5c112.png">

这个就是随机从无标训练数据中取N个构成一个branch，对于branch里的任意图像，通过上述方法构造positive case，形成两个图像增强视图:

1.Aug1/Aug2

Aug1 和Aug2各自包含N个增强数据，并分别经过上下两个分枝，对增强图像做非线性变换，这两个分枝就是SimCLR设计出的表示学习所需的投影函数，负责将图像数据投影到某个表示空间。

<img width="870" alt="image" src="https://user-images.githubusercontent.com/40928887/127969045-853761da-461b-4a5e-ac3c-f32a7ad6e5f4.png">

**为什么我们要进行两次非线性变换呢, 第一个就是CNN的非线性，第二个就是FC的projector。具体原因后文说，为什么不encoder之后就直接弄呢？**

## 3.2 距离度量
<img width="858" alt="image" src="https://user-images.githubusercontent.com/40928887/127969216-bc42ed39-2fde-4e38-bc8b-f84e33172a57.png">
这个就是对表示向量L2正则后的点积或者表示向量间的cosine相似性。
<img width="288" alt="image" src="https://user-images.githubusercontent.com/40928887/127969267-165681da-8cd7-4307-b1d7-28944cc0e25b.png">

<img width="864" alt="image" src="https://user-images.githubusercontent.com/40928887/127969281-3fb1e73f-6257-436c-9843-a132a31b5229.png">
<img width="871" alt="image" src="https://user-images.githubusercontent.com/40928887/127969361-33198022-38b5-43c4-9d98-072603c57ded.png">

这里的损失函数就是InfoNCE，也就是希望正例的相似度越高越好，也就是在表示空间内距离越近越好;分母部分，就是鼓励负例之间的向量相似度越低越好，也就是距离越远越好。

## 3.3 本身就是pretrian
本身这个过程，其实是标准的预训练模式；利用海量的无标注图像数据，根据对比学习指导原则，学习出好的Encoder模型以及它对应产生的特征表示。

>好的encoder，就是输入图像能够学会并且提取出关键特征，这个过程跟BERT的MLM自监督预训练其实目的相同，只是做法上有差异。

**Encoder学好后,可以在解决下游具体任务的时候，用学习到的参数初始化resnet，用下游任务标注数据来Fine-tuning模型参数，期待预训练阶段学到的知识对下游任务有迁移作用。**

>SimCLR看着有很多构件，比如Encoder、Projector、图像增强、InfoNCE损失函数，其实我们最后要的，只是Encoder，而其它所有构件以及损失函数，只是用于训练出高质量Encoder的辅助结构。目前所有对比学习模型都是如此，这点还请注意。

**我们的对比学习结构目前都是这样，有很多下游不需要的辅助结构**

## 3.4 为什么需要两次非线性映射，也就是咱们这里的projector到底是什么作用？
其实MoCo的那里没有使用encoder这里才是一个符合直觉的做法，projector是在后续的SimCLR在实验中验证的，加上这个projector的话对于提升模型效果改进非常明显，这个从经验角度说明两次投影变换是必须的。

>这个差异就是
>1.Encoder后的特征表示会有更多包含图像增强信息在内的细节特征，**而这些细节信息经过projector之后，很多都被过滤掉了。**
>目前这里两次就是实验结果，没有理论验证。

**我们可以猜测一个理论验证:**
我们知道，一般的特征抽取器，在做特征提取的时候，底层偏向抽取通用的低层特征，往往与任务无关，通用性强；接近比如分类任务的高层网络结构，更倾向编码任务相关的高阶特征信息。

>那么我们的encoder和projector也应该是如此，在做特征提取的时候，底层往往偏向抽取通用的低层特征，往往这个与任务无关，通用性强。

>接近比如分类任务的高层网络结构，更倾向编码任务相关的高阶特征信息。

**我们的高层projector会编码更多对比学习任务相关的信息，然后对下游任务而言这种对比学习相关的特征可能带来负面影响**

<img width="847" alt="image" src="https://user-images.githubusercontent.com/40928887/127973126-b2035d3b-b143-48e2-9806-b7f22ef3fc62.png">

## 3.5 SimCLR最大的贡献
1.一个是证明了复合图像增强很重要；

2.这个Projector结构。

**两者的结合给对比学习系统带来很大的性能提升，将对比学习性能提升到或者超过了有监督模型，在此之后的对比学习模型，基本都采取了encoder+projector的两次映射结构，以及复合图像增强的方法。**

<img width="861" alt="image" src="https://user-images.githubusercontent.com/40928887/127973330-214f175b-092b-445c-8a71-842b321078e0.png">

**这个就是典型的负例对比学习系统**

# 4.对比学习到底在干什么@hypersphere
<img width="829" alt="image" src="https://user-images.githubusercontent.com/40928887/127973404-56dfd2d0-f72d-4e75-aeeb-8b18726c1e5f.png">

对比学习在做特征表示相似性计算时，要先对表示向量做L2正则，之后再做点积计算，或者直接采用Cosine相似性，为什么要这么做呢？

<img width="844" alt="image" src="https://user-images.githubusercontent.com/40928887/127973470-1254deab-8667-4724-af79-898d972c8150.png">

**我们这里使用单位超球面很有意义**

<img width="860" alt="image" src="https://user-images.githubusercontent.com/40928887/127973506-51952792-3457-41a3-a622-234df9fabddd.png">

所谓“Alignment”，指的是相似的例子，也就是正例，映射到单位超球面后，应该有接近的特征，也即是说，在超球面上距离比较近；所谓“Uniformity”，指的是系统应该倾向在特征里保留尽可能多的信息，这等价于使得映射到单位超球面的特征，尽可能均匀地分布在球面上，分布得越均匀，意味着保留的信息越充分。乍一看不好理解“分布均匀和保留信息”两者之间的关联，其实道理很简单：**分布均匀意味着两两有差异，也意味着各自保有独有信息，这代表信息保留充分。**

相似的feat这个两两有差异的思想蛮好的。

# 5.model collapse
<img width="818" alt="image" src="https://user-images.githubusercontent.com/40928887/127973925-0386b9a6-1cac-4f0f-b24c-a0a500a38213.png">

*Uniformity特性的极端反例就是所有数据都映射到单位超球面同一个点上，这个就是极度违背了Uniformity。*

>这个就是代表所有数据的信息都被丢失了，体现为数据极度不均匀的分布到了超平面同一个点上。所有数据经过特征表示映射过程 [公式] 后，都收敛到了同一个常数解，一般将这种异常情况称为模型坍塌（Collapse）。如果对比学习的损失函数定义不好，非常容易出现模型坍塌的情形（参考上图）。
<img width="858" alt="image" src="https://user-images.githubusercontent.com/40928887/127974131-af9faa26-cc28-40ce-91b2-7ef96667cc48.png">

**像InfoNCE的分母就是利用uniformity的属性来让特征尽可能均匀得分布在这个单位超平面上，保留了尽可能多的有用信息。**

所有在损失函数中采用负例的对比学习方法，都是靠负例的Uniformity特性，来防止模型坍塌的，这包括SimCLR系列及Moco系列等很多典型对比学习模型。


<img width="867" alt="image" src="https://user-images.githubusercontent.com/40928887/127974283-0488f124-8e0d-44da-8e22-1c60cbdaf67f.png">

# 6.InfoNCE的temperature scalar的解析
对比学习模型要想效果比较好，温度超参 [公式] 要设置一个比较小的数值，一般设置为0.1或者0.2。

<img width="845" alt="image" src="https://user-images.githubusercontent.com/40928887/127974368-97c7dc65-6880-43a7-a98c-8f31b918ddf8.png">

**这个就是表明InfoNCE是一个能够感知负例难度的损失函数，之所以能够做到这点主要就是依赖超参数。**

<img width="860" alt="image" src="https://user-images.githubusercontent.com/40928887/127974508-585e65ca-e83e-4a09-887b-4bd06a85a513.png">
<img width="848" alt="image" src="https://user-images.githubusercontent.com/40928887/127974596-52943e72-0a2b-4b5d-ad2e-b2114777c9c9.png">

这里是说温度超参越小这个分布就会越均匀。

<img width="869" alt="image" src="https://user-images.githubusercontent.com/40928887/127974653-a3281330-5c3b-4610-9623-a91c2f44de00.png">
<img width="864" alt="image" src="https://user-images.githubusercontent.com/40928887/127974740-199e4386-988b-4b08-b199-12da45e34121.png">

**我们这个温度也应该找到一个很好的平衡点，不然就是破坏了类内的信息啦**

https://zhuanlan.zhihu.com/p/367290573











