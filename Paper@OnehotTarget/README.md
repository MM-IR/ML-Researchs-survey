# 1.AAAI2021: Label Confusion Learning to Enhance Text Classification Models
>Representing a true label as a one-hot vector is a common practice in training text classification models.

## 1.1 Motivation@只是源于自己的一个思考而已
>one-hot representation may not adequately reflect the relation between the instances and labels.
>因为labels are often not completely independent and instances may relate to multiple labels in practice.

这个不合适的one-hot representations tend to train the model to be over-confident.
**这个就是不适合confused datasets，dataset with very similar labels或者noisy数据集。**

>虽然label smoothing can ease this problem in some degree, it still fails to capture the realistic relation among labels.

## 我们的创新点summary
>1.我们提出一个novel Label Confusion Model as an enhancement component to current popular text文本分类模型。

<img width="499" alt="image" src="https://user-images.githubusercontent.com/40928887/124877466-af8e7180-dffd-11eb-91b3-ff4aa13b0ded.png">

2.Further experiments also verify that LCM is especially helpful for confused or noisy datasets and superior to the label smoothing method.

## Label Smoothing
>这个就是一个超参数, 然后其余的都是(1-x)/(n-1).

## 2.我们指出的learning paradigm的两个缺陷
1) one-hot label representation is based on the assumption that all categories are independent with each other.

>但是在real scenarios, labels往往不是independent的，而且instances may relate to multiple labels。尤其是针对那些confused dataset。

- 简单represent the true label通过这个onehot vector fails to 吧这个relation between instances以及labels into account。

**这个就是进一步限制了这个learning ability。**

2) large annotated data对于DL的成功很重要，但是inevitable我们标注的有误。

### 我们将这个问题统一归类为label confusion problem.
>label smoothing method is proposed to remedy the inefficiency of onehot vector labeling.
>但是这个还是不能capture 这个realistic relation among labels，因此不足够去solve这个问题。

## 我们的LCM的大概
>1.这个就是learn the representation of labels以及计算这个相似性semantic with 输入text表达来评估他们的dependency。
>这个就是然后transfer 到label confusion distribution。LCD。

**在这之后, 我们的原始的one-hot label vector is added to the LCD with a controlling parameter and normalized by a softmax function to generate a simulated label distribution.**

我们就是使用SLD来替代onehot。

## 我们核心的想法就是a label distribution that can reflect the similarity can reflect the similarity relation between labels ->可以帮助咱们训练一个更加强大的模型，
>最常见最直觉的想法就是calculate the similarity between every two labels.
>我们就是直接用这个normalized similarity values来supervise这个模型的学习。

**但是这个情况，就是一个很大的问题: label distributions got in this way are all the same for instances with the same label, regardless of their content.**

这个就是即使两个instances有相同的label，但是事实上他们的content也是不一样的，所以他们的label distirbution就应该是自适应以及不同的。

<img width="886" alt="image" src="https://user-images.githubusercontent.com/40928887/124882129-a5bb3d00-e002-11eb-8e2b-7aaa91b43833.png">

>因此我们应该construct这个label distribution using the relations between instances and labels, thus the label distribution will dynamically be changing for different instances with the same label.

# 3.技术解析
左边的网络模块就是最一般的分类模块，右边的那个模块就是一个Label Confusion Model.
<img width="688" alt="image" src="https://user-images.githubusercontent.com/40928887/124882491-021e5c80-e003-11eb-85e4-e9f9670ed0f3.png">

>LCM就是由两个部分来组成的。

1.label encoder;@DNN to generate the label representation matrix

2.simulated label distribution computing block.@similarity layer and a SLD computing layer.

>这个就是首先把label的表达获得到，然后就是拿这个和instance representation进行相似性计算@dot product，然后就是一个nn with softmax就是得到这个label confusion disrtibution.

**Thereby, 这种就是一个dynamic, instance-dependent distribution, 这个就是superior to the distribution that solely considers the similarty among labels, or simply a uniform noise idistribution like the way is LS**
<img width="408" alt="image" src="https://user-images.githubusercontent.com/40928887/124891428-8ecd1880-e00b-11eb-977e-f0a354f634e4.png">
**这边咱们重视的是下面的这个情况～**
<img width="446" alt="image" src="https://user-images.githubusercontent.com/40928887/124891398-8aa0fb00-e00b-11eb-963e-70e71e05a11c.png">

我们用一个parameter来表示原始onehot的占比。

<img width="424" alt="image" src="https://user-images.githubusercontent.com/40928887/124891573-af956e00-e00b-11eb-9ad7-4729326b115d.png">

这个一般就是选择的4.

## 实验分析
1.为什么我们的LCM-based 分类模型可以取得比较好的结果:

1)LCM part就是学习simulated label distribution SLD during training which captures the complex relation among labels by considering the semantic distribution;
这个就是superior to simply using one-hot，

2)这里i 就是肯定是存在一些mislabeled data的，especially for datasets with大量的categories或者very similar labels。

**在这个情况下，training with one-hot label representation tend to be influenced by these mislabeled data more severely.**
但是如果咱们使用SLD的话，value on the index of the wrong label will be crippled and allocated to those similar labels.

>因此这个misleading of the wrong label就是relatively trivial。

3)除了mislabeled data,这个given label有一些相似性的时候，it is natural and reasonable to label a text sample with a label distribution that tells various aspects of information.

>但是现有的就没有这么搞过。

<img width="431" alt="image" src="https://user-images.githubusercontent.com/40928887/124893070-00599680-e00d-11eb-9c55-4f1b6d88d970.png">







