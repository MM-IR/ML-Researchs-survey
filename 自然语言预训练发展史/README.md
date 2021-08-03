# Bert值得这么高的评价吗？我个人判断是值得。
那为什么会有这么高的评价呢？是因为它有重大的理论或者模型创新吗？其实并没有，从模型创新角度看一般，创新不算大。但是架不住效果太好了，基本刷新了很多NLP的任务的最好性能，有些任务还被刷爆了，这个才是关键。另外一点是Bert具备广泛的通用性，就是说绝大部分NLP任务都可以采用类似的两阶段模式直接去提升效果，这个第二关键。客观的说，把Bert当做最近两年NLP重大进展的集大成者更符合事实。

>Bert可以当作是近两年NLP重大进展的集大成者

<img width="667" alt="image" src="https://user-images.githubusercontent.com/40928887/127975102-be5b4033-6e5d-42d9-bf58-52aab7ba9e77.png">

# 1.首先是图像的pretraining
自从深度学习火起来后，预训练过程就是做图像或者视频领域的一种比较常规的做法，有比较长的历史了，而且这种做法很有效，能明显促进应用的效果。
<img width="635" alt="image" src="https://user-images.githubusercontent.com/40928887/127975146-19b662e4-241d-497d-b417-6be573b12c9c.png">

<img width="677" alt="image" src="https://user-images.githubusercontent.com/40928887/127975215-a4ef7b3f-4519-4797-9f2c-f9d5ddd2dc8f.png">
<img width="677" alt="image" src="https://user-images.githubusercontent.com/40928887/127975243-0ea4495e-0c82-4a47-8bf8-a63b4e3b54a9.png">

## 1.1 为什么这种预训练的思路是可行的？
<img width="662" alt="image" src="https://user-images.githubusercontent.com/40928887/127975299-4a2a808d-5ecb-470d-aacb-ed3b2ef66709.png">
对于层级的CNN结构来说，不同层级的神经元学习到了不同类型的图像特征，由底向上特征形成层级结构

<img width="675" alt="image" src="https://user-images.githubusercontent.com/40928887/127975393-6a334e31-3d2a-4ebe-8b03-a5cf2b853895.png">

**核心想法就是底层的网络参数抽取出特征跟具体任务越无关，越具备任务的通用性，所以这个也是为何一般用底层预训练好的参数初始化新任务网络参数的原因，而高层特征跟任务关联较大，实际上可以不使用。**

>我们高层的feat这里的话就是跟任务关联较大，实际可以不用使用，或者采用fine-tuing用新数据集合清洗掉高层无关的feat extractor。

<img width="624" alt="image" src="https://user-images.githubusercontent.com/40928887/127975639-83c2965d-16c9-4c50-80c0-39e8b95b4e8f.png">
<img width="684" alt="image" src="https://user-images.githubusercontent.com/40928887/127975689-97ddc22f-5f00-45fc-bb34-11a9a94aff87.png">

那么NLP干什么不做？人家并不比你傻。

>其实也不是说word embedding不成功，一般加到下游任务里，都能有1-2个点的性能提升，只是没有那么耀眼的成功而已。

# 2.word embedding考古史
<img width="674" alt="image" src="https://user-images.githubusercontent.com/40928887/127975840-41a14fd1-de1b-4940-84d6-c63e28a7dbe6.png">

这里一般使用语言模型来做预训练
<img width="524" alt="image" src="https://user-images.githubusercontent.com/40928887/127975946-ac4f4986-3728-43bf-b55b-71378d1599cf.png">
<img width="684" alt="image" src="https://user-images.githubusercontent.com/40928887/127975998-865c2e02-5bf0-474f-87d2-2029a47f35a3.png">
<img width="678" alt="image" src="https://user-images.githubusercontent.com/40928887/127976034-a078dabf-2686-4be5-a556-c5195ea5a290.png">
<img width="683" alt="image" src="https://user-images.githubusercontent.com/40928887/127976056-59d7f87b-38ca-47db-919e-ebd8ba2e16cf.png">

**这个其实就是pretraining，和咱们利用图像的low-level feat一个道理**

区别无非Word Embedding只能初始化第一层网络参数，再高层的参数就无能为力了。下游NLP任务在使用Word Embedding的时候也类似图像有两种做法，一种是Frozen，就是Word Embedding那层网络参数固定不动；另外一种是Fine-Tuning，就是Word Embedding这层参数使用新的训练集合训练也需要跟着训练过程更新掉。

## 2.1 为什么Word2Vec这种形式在18年以前虽然一直都用，但是帮助没有大到闪瞎狗眼呢？
<img width="598" alt="image" src="https://user-images.githubusercontent.com/40928887/127977188-e36c7cd9-2541-4068-8fcb-1056a58405e3.png">

>一个很大的问题就是多义词问题，这个就是两个含义: 比如多义词Bank，有两个常用含义，但是Word Embedding在对bank这个单词进行编码的时候，是区分不开这两个含义的，因为它们尽管上下文环境中出现的单词不同，但是在用语言模型训练的时候，不论什么上下文的句子经过word2vec，都是预测相同的单词bank，而同一个单词占的是同一行的参数空间，这导致两种不同的上下文信息都会编码到相同的word embedding空间里去。所以word embedding无法区分多义词的不同语义，这就是它的一个比较严重的问题。

**两种不同的上下文信息都会编码到相同的word embedding空间里去(one line)。。。所以这个无法区分多义词的不同语义。**

# 然后ElMo给出一种解决多义词的方案。@这里有利用到当前情况下的上下文信息
这个精髓就是deep contextualization, 这个context更加关键，**之前的word embedding本质上是static的方式，这个就是训练好之后每个单词的表达就固定住了，以后使用的时候，不管新句子上下文单词是什么，这个单词的word embeding不会跟着上下文场景而变化。**

>ELMo这个就是我事先用LM学习一个单词的word embedding，这个时候多义词无法区分，不过这个没关系，在我实际使用word embedding的时候，单词已经具备特定上下文了。

**我可以根据上下文单词的语义去调整单词的Word Embedding表示**

这样经过调整后的Word Embedding更能表达在这个上下文中的具体含义，自然也就解决了多义词的问题了

<img width="650" alt="image" src="https://user-images.githubusercontent.com/40928887/127983062-c21aed0c-90c6-46b2-9df2-cbac47ce9885.png">
<img width="704" alt="image" src="https://user-images.githubusercontent.com/40928887/127983112-f6d7d4df-16ad-4fe5-a7d5-e613f2d6bf46.png">
<img width="679" alt="image" src="https://user-images.githubusercontent.com/40928887/127983146-f8f90b21-4d7d-405c-b4c1-053e3d774d2f.png">

**这个就是一定程度上可以区分多义词的效果，你想一想其实可以明白原因。**

<img width="581" alt="image" src="https://user-images.githubusercontent.com/40928887/127983202-2179af27-a57c-472d-adc8-7d6680c32229.png">
<img width="686" alt="image" src="https://user-images.githubusercontent.com/40928887/127983318-c57bb974-87af-4ad2-9948-112f0c6444c4.png">
<img width="683" alt="image" src="https://user-images.githubusercontent.com/40928887/127983340-c2aedec3-397d-4903-9c4f-147aa6ef61ce.png">
<img width="692" alt="image" src="https://user-images.githubusercontent.com/40928887/127983365-4df24d53-85fa-441f-8ce0-9c4a4b23d8c8.png">

但是ELMo也有一些缺陷
<img width="447" alt="image" src="https://user-images.githubusercontent.com/40928887/127983389-445c082c-2697-4d5a-81ba-1b3da63a5365.png">


