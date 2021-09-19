>Freman: If 

<img width="528" alt="image" src="https://user-images.githubusercontent.com/40928887/133914346-f9ed5f5a-8fa9-473f-9054-2622b118d3b8.png">

这个就是生成的话也是需要足够的理解才可以创造的。

# 1.PixelRNN
<img width="523" alt="image" src="https://user-images.githubusercontent.com/40928887/133914370-20b7756a-aaa1-4a65-a2aa-6e8f4c1ce31e.png">

VAE是2013.

<img width="552" alt="image" src="https://user-images.githubusercontent.com/40928887/133914409-c7ccb2ec-dfbf-4c32-a6d9-fe840a4b3350.png">

这个就是我们可以使用RNN来output这个pixels。3d vector输入输出。

**一行一行来～unsupervised**

<img width="522" alt="image" src="https://user-images.githubusercontent.com/40928887/133914448-ccdeca36-1f5c-43b4-a3a7-189474637112.png">

**比如这种狗的下半身的/**

>除了image，其实还有不少的应用。

比如WaveNet和VideoPixel
<img width="521" alt="image" src="https://user-images.githubusercontent.com/40928887/133914485-b5407cdb-fd56-4a1e-9e45-6481543b4e27.png">

## create new pokemon
<img width="537" alt="image" src="https://user-images.githubusercontent.com/40928887/133914510-deec862a-43fa-4b98-b4bd-26f4af256525.png">

这个就是使用sigmoid往往都是落在中间，很难有这个极端的value。

<img width="523" alt="image" src="https://user-images.githubusercontent.com/40928887/133914552-3e9fecbc-4834-437b-976b-e7fdf10502d7.png">

所以我们就是使用cluster+onehot来表示这个一个pixel的颜色。

<img width="555" alt="image" src="https://user-images.githubusercontent.com/40928887/133914561-d5501d5f-33ae-4ed7-bf35-d6774c4df886.png">

我们用1-layer LSTM with 512 cells.

<img width="545" alt="image" src="https://user-images.githubusercontent.com/40928887/133914581-64cfb439-1638-4aef-818d-64376660694e.png">

所谓的创造其实是无法evaluate的。

<img width="557" alt="image" src="https://user-images.githubusercontent.com/40928887/133914628-42bcf70e-ee7d-4bc5-81d5-37a099c19c04.png">

当然很多看不懂啦。


# 3.VAE
比如AE
<img width="522" alt="image" src="https://user-images.githubusercontent.com/40928887/133914639-a166e82d-a90f-48e7-93ce-48eb34192779.png">

这个就是随机的code生成的image往往效果不是那么好。

那么就有了改进: VAE(Variational)

<img width="562" alt="image" src="https://user-images.githubusercontent.com/40928887/133914670-cb682fd7-2998-4998-8c48-fdd554dd590c.png">

VAE这里其实比较神奇，我们生成的这两个vetcor+这个normal distribution的做的生成的vector，然后就是除了minimize之前的reconstruction error，我们还minimize一个奇怪的东西。

OpenAI的VAE
<img width="555" alt="image" src="https://user-images.githubusercontent.com/40928887/133914706-fe4fa321-7bef-4757-a0af-c2ddf7b9c01a.png">

>VAE其实不太清楚，不过理论上你可以控制你想要生成的image。

<img width="550" alt="image" src="https://user-images.githubusercontent.com/40928887/133914741-4fe35ba1-83b8-4373-8c6c-9593cca027f0.png">

**而且这个可以找到每一个dim到底代表什么意思。**
<img width="463" alt="image" src="https://user-images.githubusercontent.com/40928887/133914749-f1041ec2-89c6-49e2-b27d-4d039bb2998c.png">

这个就是可能每个dim确实有一定意义。

看起来其实还是有一定的意义的。

**有的人还可以用VAE来写诗**

<img width="552" alt="image" src="https://user-images.githubusercontent.com/40928887/133914816-ce64c263-967f-48b6-b912-ebb0f570f504.png">

这个也是🈯不同的code space啦。

# 4.why VAE？
>1.Intuitive Reason

<img width="551" alt="image" src="https://user-images.githubusercontent.com/40928887/133914865-08bb6105-ec9a-438c-82a0-73e940145700.png">

以前对于一张满月可能就是一个点会对应到上面。

现在我们就是VAE这里就是我们在这个code上加一些noise，一定范围都可以返回到这个满月。然后肯定这个会有重叠的点，那么这个重叠的地方就是过渡状态。

这个就是模型自己的encoder自己去决定这个noise的variance到底应该多大。

**这个variance和unifrom乘一下就是比较广泛的范围啦。我们不可以让这个variance变成0，所以下面我们要进行调整。**
<img width="551" alt="image" src="https://user-images.githubusercontent.com/40928887/133915267-56dfcb91-9b13-4bce-a98a-3581bd896706.png">

绿色的这条线就是minimization。**我们不像**




