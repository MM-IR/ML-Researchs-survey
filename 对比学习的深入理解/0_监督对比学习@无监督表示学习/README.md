# 1.self-supervised contrastive learning
>这个就是要求训练过程中不使用图片包含的GT类别信息(不能使用分类任务来训练)，那么我们该如何训练backbone network，来为每张图片来提出好的feat？

**这个答案有很多，我们这里就是介绍: 自监督对比学习，这个核心就是如何给数据自动产生一种标签，然后使用该标签来进行某种监督学习。**

>1.比如，无标签的图片，可以吧图片随机旋转一个角度alpha，比如90/180/270，然后用旋转后的图片作为输入，训练网络来预测到底旋转了多少角度。

<img width="674" alt="image" src="https://user-images.githubusercontent.com/40928887/127961756-dede348f-14b9-4c1e-933e-a6246e2978ca.png">

**我们这里就是分别随机对每张图片进行两次data augmentation，然后就是来源于同一张图片的feat越接近越好。**

<img width="666" alt="image" src="https://user-images.githubusercontent.com/40928887/127961805-f93d0850-f916-4c25-a5ef-c30d7e490117.png">
<img width="683" alt="image" src="https://user-images.githubusercontent.com/40928887/127961812-e120bff3-9550-4c39-b713-3828238edbda.png">

**但这种自监督的对比学习来学习图像表示的时候有一个不足: 我们没有考虑到属于同一个类的不同图片之间的feat的相关性。**
<img width="647" alt="image" src="https://user-images.githubusercontent.com/40928887/127961970-b14f8afc-13dd-4f2f-9bed-9d3870a173c3.png">
<img width="678" alt="image" src="https://user-images.githubusercontent.com/40928887/127961981-bc02d055-40b1-4f1b-aef4-e1fff54a7772.png">

>直觉上同类图片的feat。其实也应该越近越好。但是由于自监督对比学习的设定里面不使用图片所属的类别信息，所以我们无法知道哪些图片属于同一类，因此**无法让同类图片之间的feat彼此距离相近。**

>如果能使用图片的类别 label信息，是否能提高以上“自监督对比学习“的feature的质量？

# 2.监督对比学习
<img width="674" alt="image" src="https://user-images.githubusercontent.com/40928887/127962089-e8fff2ee-1738-4373-8bf3-1b0d85143f9c.png">
<img width="679" alt="image" src="https://user-images.githubusercontent.com/40928887/127962109-6e431857-2de8-4fb5-a924-5d90d1613521.png">
<img width="673" alt="image" src="https://user-images.githubusercontent.com/40928887/127962120-cc391ff0-de9e-4b03-a2df-cc844fb862ab.png">

这个就是同类别的图片的feat，在超球面上的距离，很近。

**这个其实也是替代交叉熵的一个很好的方法**
