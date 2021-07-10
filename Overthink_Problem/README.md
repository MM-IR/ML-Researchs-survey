<img width="663" alt="image" src="https://user-images.githubusercontent.com/40928887/125154327-1c377680-e18c-11eb-92d2-440f9e7459e3.png">
# 1.abstract
>We characterize a prevalent weakness of deep neural networks(DNN) - overthinking.描述xxx的一个缺点。

这个就是DNN可以reach正确的predictions before最后的layer。

>overthinking: 这个是计算上的wasteful的，it can also be destructive when, by the final layer, a correct prediction changes into a misclassification.

<img width="272" alt="image" src="https://user-images.githubusercontent.com/40928887/125154787-71748780-e18e-11eb-8ab7-3273eba6a9ed.png">

## For prediction transparency,
我们这里就是提出了一个Shallow-Deep network, a generic modification to off-the-shelf DNNs that introduces internal classifiers.

**这里就是四个modern architectures，在三个任务上训练，to characterize the overthinking problem.**


