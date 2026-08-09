# 3 Tensor Core 从 mma.sync 到 tcgen05 —— 完整讲课转录

## 视频信息

- 标题：北京大学未名超算队 × LCPU AI Infra Seminars 系列讲座：3-Tensor Core 从 mma.sync 到 tcgen05
- 链接：https://www.bilibili.com/video/BV1LxM96eE43
- UP 主：北京大学Linux俱乐部
- 主讲人：孙远航（B 站字幕音译作"孙晓航"，以简介为准；21 届本科，曾任社团社长与队长）
- 时长：2:40:11
- 活动背景：2026 年暑期 Weiming HPC Training Camp × LCPU AI Infra Seminars 第四次活动录播

## 校订说明

本转录以 B 站 ai-zh 自动字幕为底稿，按语义校订整理（该视频无 ai-en 字幕）。视频为中文讲授，正文沿用中文。ai-zh 字幕覆盖全片（末条时间戳 02:40:11.040，视频时长 2:40:11），未重新运行 ASR。原始带时间戳字幕保留在 `raw/lectures/captions/`；英文技术名词（mma.sync、wgmma、tcgen05、TMEM、mbarrier、ldmatrix、swizzle、descriptor、proxy 等）按上下文恢复，个别听辨不清处已在正文与附录标注。

---

## 开场：今天要解决的两个核心问题

今天要聊的话题是 tensor core。有一个说法：看一个人会不会写 kernel，就看两件事——第一，他是不是会用 shared memory；第二，他是不是会用 tensor core。今天这两块我们都会涉及。

今天要解决两个核心问题：

1. **tensor core 到底是什么**——我们应该怎么去认识它；
2. **我们到底要怎么为 tensor core 准备数据**——因为我们需要把要算的东西"喂"给 tensor core，希望这个过程符合底层硬件的需要，同时能够和 CUDA 的库以及整个内存 hierarchy 做好配合。

tensor core 在每一代硬件之间是有变化的：A100、H100 和 B200 这三大硬件上，tensor core 有一些变化，但里面的概念是从最早开始一贯往后的。我们会按这三代硬件去讲它到底是什么、应该怎么用，同时探索一下加速器硬件是怎么设计、怎么使用的。

最后会讲低精度与 tensor core 的结合：最早大家都是用 FP32 做训练，后来大家突然觉得精度可能不需要那么多、但需要更大的参数量，那么怎么把低精度和 tensor core 结合好、把计算加速，同时又保证训练效果不受到比较大的影响。

需要说明：这一讲只讲 tensor core 本身，tiling 和 pipeline 那套东西会在下一场讲。所以大家听的时候可能会觉得"它好像还差点什么"——确实还差一点，但这一节课的内容本身差不多要三个小时了，可能覆盖不到，接下来我们会讲得比较多。

## 一、为什么需要 tensor core

### 计算强度问题

之前在课上写过一个 FP32 的 GEMM，计算强度很低：它循环 K 维，每次读 A 和 B、做一次乘加、写回，整个计算强度只有 0.25 FLOP/byte。这就很令人困惑：数据读取这块压力很大。

tensor core 想解决的问题就是：我们读进去一些东西的时候，尽量能够多算一些，让读的压力减小。如果你想打满一张卡——比如把 A100 打满——它的 FP32 tensor core 平衡点是 9.75 FLOP/byte。9.75 是怎么算出来的：整张卡 FP32 tensor core 算力上限是 19.5 TFLOPS，带宽 2 TB/s，要打满卡，每读进去 1 字节就要算差不多 10 次计算。

而且这还不够：这个要求和 CPU 其实差得不远。想要做得更好，就要找一套方法提高计算强度。让数据读取变得更快很难，但我们可以想办法让每次读取多算一些——这就是 tensor core 最主要的设计思想。

### 为什么矩阵乘适合硬件加速

第一个问题：为什么矩阵乘比较适合硬件加速？

- 它在做大模型、以及以前的机器学习时是一个非常重要的计算形态。比如很早 Yann LeCun 做的手写数字识别（一个全连接的东西），就可以用矩阵乘比较好地表达。
- 它有"标号性质"：计算量是 $O(N^3)$，读取量是 $O(N^2)$。只要把 N 扩大，数据读取相对于计算会变少，整体计算强度就能提高，就可以把更多信息放到更大的参空间里去训练、做更好的提取。

各种加速设备并不是 tensor core 第一个做这件事的。比如 Intel 的 Xe 也试图通过 SIMD 之类的机制去做加速，本质上也是在做一个类似的矩阵乘计算。但最后 NVIDIA 的 tensor core 胜出了，原因有很多，我觉得比较重要的一条是：**tensor core 和 CUDA 的 SIMT 结合得很紧密**。用 CUDA 编程时本来要学的那些加速计算的知识依然存在，通过这种结合能表达出比较好、比较有趣的性质。

### 为什么硬件不直接算"整个 tensor"

那为什么硬件不直接做成"传入两个完整的 tensor，把结果算好返回"？比如 PyTorch 里放两个 tensor、中间放一个乘法。其实也有人这么做——有些 NPU 就是干这个事的。但这样的设备不灵活：它只能表达 tensor 计算，而我们有大量计算不能表达成 tensor。为了把"不能表达成 tensor 的计算"和"要用 tensor core 的计算"结合好，就需要在现有设备上做设计。

所以 tensor core 的定义是：**一个硬件用来算 A×B+C 的单元**，A、B、C 的形状是固定的，就是一小块——比如很常见的 16×8×16 这样的 MNK 小块。如果做成整个向量进出，中间的 cuBLAS 之类库会把它隐藏起来；而把能力拆解到现有的 SIMT 硬件上之后，tensor core 除了自己发挥好作用，CUDA core 也能发挥好作用，整个 GPU 就活了。

## 二、mma.sync：一条 tensor core 指令长什么样

举一个非常直接的例子，一条 SASS 指令，对应 PTX 的 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`：

- **MMA**：matrix multiply and accumulate，除了乘还有累加；
- **sync**：在跑这条 tensor core 指令之前有一次隐式的 warp 同步；
- **aligned**：同一个 warp 里的 32 个线程必须同时执行这条指令，否则是未定义行为；
- **shape M16N8K16**：A 是 16×16（M×K），B 是 8×16（N×K），C/D 是 16×8；
- **row/col**：A 是行主序（K 维连续），B 是列主序（也是 K 维连续）——A 和 B 都是 K 维连续的排布；
- **数据类型**：D 和 C 是 FP32，A 和 B 是 FP16（16 位浮点），乘完之后用 32 位累加。

为什么 D 是 FP32？A、B 是 FP16，它们的乘积如果还用 FP16 表示，精度会爆、范围也不够。虽然输入用低精度，但最后的结果是高精度的 FP32 矩阵。

怎么知道有哪些指令可以跑？最好的办法是看 NVIDIA 的 PTX 文档（课前已经发给大家了）：看 mma 那一节，它会列出有哪些形式、如何组合——a layout、b layout、d type、shape、c/d type，以及每一种指令需要多高的 compute capability 支持。看什么材料都不如看这份文档清楚。

### Compute capability

每一代 GPU 的 feature 不一样，一个程序需要知道自己跑在什么卡上，才能把底层硬件利用好。每一代卡都有自己的 compute capability，由两部分组成：8.0、9.0、10.0 等。前面的 major（8/9/10）表示这一代的大 feature，同一代内大 feature 基本是兼容的：8 这一代主要是 Ampere（A100），9 是 Hopper（H100），10 是 Blackwell（B200）。同一代内用第二位区分具体卡，比如 A100 是 8.0，RTX 4090 是 8.9。每一代都有它的特点，后面会对应讲 tensor core 的特点。

另外提一下：硬件不支持完全 32 位的 tensor core，最多支持到 TF32，原因是为了节省电路——电路越多成本越高、卡越贵、功耗越高，所以希望大家尽量节省电路，把电路放在别的地方。

### 计算题：M16N8K16 指令的计算强度

算一条 M16N8K16 指令的计算强度：读多少字节、算多少次计算。

矩阵乘的本质是三重 for 循环（for m, n, k），对 K 方向做乘加。计算量是 $MNK$ 次乘加、每次乘加算 2 次浮点操作，所以是 $2 \times 16 \times 8 \times 16 = 4096$ FLOP。读 A、B，写回 D（这里没有读 C），总字节数是

$$16\times16\times2 + 8\times16\times2 + 16\times8\times4 = 512+256+512 = 1280\ \text{bytes}$$

所以强度是 $4096/1280 \approx 3.2$ FLOP/byte，比纯 CUDA core 的写法强度高了约 26 倍。也就是说，读进来一段 A 和 B 之后，做的乘法多了非常多。

一般公式是：强度 $= 2MNK/(MK+NK+MN)$（每个元素按 4 字节计）。要进一步提高强度，就需要把 MNK 扩大——虽然读的东西多了一点，但强度增加比读取量增加快得多。所以之后每一代 tensor core 都在试图增大 MNK。但增大了之后，数据输入压力也会随之提高，而且"算出来的东西有地方放"（需要用寄存器存下来）——这就是后面几代演进的直接动因。

## 三、A100 的 FP16 tensor core 峰值算力

来算一下 A100 的 FP16 tensor core 峰值：

- 刚刚的 mma.m16n8k16 在底层会发一条 HMMA 指令，查白皮书可知这条指令延迟是 8 个 cycle；
- A100 每个 SM 有 4 个子分区（sub-partition），每个子分区有一个 tensor core，每个 warp 在其中一个子分区上跑，同时可以有 4 个子分区跑 tensor core 指令；
- A100 一共有 108 个 SM；
- boost 频率约 1.41 GHz。

峰值 = 108（SM）× 4（子分区）× 4096（每条指令 FLOP）/ 8（cycle）× 1.41 GHz ≈ **311.9 TFLOPS**。

注意：tensor core 指令是不流水的——发出去之后要停 8 个 cycle，这期间如果你不去开多 warp，CUDA core 就会等。这个结果和 A100 手册上的 FP16 峰值（约 312 T）是一致的。虽然 4096 这个数看起来不多，但整张卡的 tensor core 算力非常恐怖，已经到了 300 多 T，普通 CPU 很难摸到这个点。

但卡的平衡点和单条 MMA 的差距还是蛮大的：卡是 312 TFLOPS，带宽只有 2 TB/s，所以要跑满 tensor core，需要达到 156 FLOP/byte 的计算强度；现在只有 3.2，远远不够。以后的课程里会有 tiling、pipeline 之类的东西尽可能做数据复用，让每次读进来的数据能被多算几次，努力逼近 tensor core 峰值。

## 四、三代 tensor core 的演进

大家可能觉得 tensor core"也就那样"：不就是 mma.sync 算一个 MNK 乘积吗？其实不是这么简单。

### A100（SM80）：全放寄存器

A100 上 A、B、C、D 全放寄存器。每个线程需要自己去算地址、取数、放到自己的寄存器里，这对寄存器文件压力很大——已经到了一百多 byte/cycle 的很高压力。

### H100（SM90）：A/B 进 shared memory，变成异步

A100 是 312 T，H100 是 990 T，翻了快三倍。如果每次还都从寄存器读这些数，寄存器带宽是不够的（寄存器是 CPU/GPU 上最贵的资源，读写灵活、带宽大、使用频率高，做不到翻倍）；而且 tensor core 在计算的时候 CUDA core 是空闲的。所以 H100 上：

- 把 A、B 从寄存器挪到 shared memory（反正本来也是从 shared memory 读进寄存器的，直接放 shared memory 更好）；
- 把 tensor core 改成**异步**（wgmma），让它和 CUDA core 异步执行——tensor core 本质上不依赖线程逐条操作，可以变成异步。

### B200（SM100）：累加器进 TMEM，单线程发射

在 B200 上想再翻倍，发现累加器 D 占用了大量寄存器；同时既然是异步的，发射也不一定要大家一块发。所以在 B 系列上引入了专门的 tensor memory（TMEM）去放 D，tensor core 与 CUDA core 的关系不再那么紧密了。

### 答疑

- **数据在寄存器还是 shared memory 是硬件决定的**：你要去适配它，不是你能决定 tensor core 和 CUDA core 的关系；但在 H 和 B 上你依然可以使用 SM80 的老指令（全放寄存器的方法）。
- **SM 结构**：一个 SM 里有 4 个小的单元，叫 sub-partition；每个 sub-partition 上有一个 warp 在跑，这个 warp 既可以用 tensor core 也可以用 CUDA core。它们通过寄存器文件和整个 SM 的 shared memory 通信。tensor core 是比较独立的一块，但数据上 CUDA core 和 tensor core 都能操作整个 SM 的数据。计算本质上是 tensor core 在算，但你要给 tensor core 背数据——在 A100 上这需要 CUDA core 帮忙。
- **会不会抢带宽**：A100 上 tensor core 是同步的，算的时候当前 CUDA core 被阻塞，所以没有抢带宽的问题（但读寄存器会有压力）；H/B 上 shared memory 带宽大部分时候打不满，所以也还好；肯定会有资源竞争问题。
- **任务类型**：tensor core 只干一件事——MMA（A×B+C=D）；CUDA core 也能算，但读算比很高、计算强度不够，最后会 bottleneck 在寄存器/shared memory 读取上。
- **为什么操作数要在寄存器上**：不是让你手动去算这条 tensor core 指令，而是你为 tensor core 准备好数据之后让它去算；把需要计算的操作数放到寄存器上是 tensor core 的硬件要求。
- **高精度**：tensor core 与高精度计算并不是无缘的，高精度计算也可以用它来模拟；但 A+C 这种没有乘法的计算没必要拿 tensor core。
- **只在 warp 级调用**：A100 上是的。写 block 时要知道每个线程属于哪个 warp：一个 warp 32 个线程，`threadIdx.x / 32` 就知道。

## 五、fragment：怎么为 tensor core 准备数据

tensor core 编程的核心就是**如何为它准备数据**，需要掌握两件事：

1. 看懂 fragment 图——一个矩阵怎么摊到每一个线程上；
2. 用好矩阵加载指令（ldmatrix）。

### 每个线程分到多少数据

mma.m16n8k16 由同一个 warp 里的 32 个线程分摊：每个线程负责 4 个 D、4 个 A、2 个 B、4 个 C（这里说的是寄存器数量）。注意 CUDA 的寄存器是 32 位的，FP16 只有 16 位，所以要把两个 FP16 打包放在一个寄存器里；FP64 则要占两个寄存器。

A 一共 256 个数（16×16），B 128 个数（8×16），C/D 各 128 个数。

### 读 fragment 图

fragment 图回答"每个线程该拿哪几个元素"。以 A 的 fragment 图为例：

- 图分四个象限（每象限 8×8），每个线程交替地拿每个小块里的数据；
- 线程编号是 warp 内的 lane id（0~31），与全局 id 无关；
- 每个线程拿 4 个寄存器（里面是 8 个 half）。比如 lane 5 拿的是：第一行的第 2、3 列和第 10、11 列……第 2、3 列两个 half 打包成一个 32 位寄存器；
- 布局是**固定**的：只有按 fragment 的方式把数据填进去，结果才是对的。为什么这么排？因为 tensor core 直接从对应的数据接口取数，这个连线方式决定了排布，与底层硬件设计相关，不是我们能控制的。做 tensor core 编程时一个核心观念是：你要做的事不是让自己写起来方便，而是让你的结构能被 tensor core 利用好——这是它的要求。

读图的方法：先确定在第几个大格子（象限）里，再确定 lane id 对应的行、列，然后算具体元素。每个标签跨两列，因为两个 16 位打包正好是一个 32 位寄存器。规律是可计算的，编程时列式子而不是数格子。

一个听众问：fragment 排布会不会有 bank conflict？这是寄存器排布，不会；但如果 A 矩阵不放在寄存器、而是从 shared memory 里取，就会有 bank conflict——后面专门讲怎么解决。

### B 的 fragment

B 是 N×K、K 方向连续的 8×16 矩阵，排布方式和 A 类似：从上往下切两份，再按 4 个线程一组划分。举例：B 的第 4 行第 4 列（坐标 4,4）在下面这一份、对应 T12 开始的一组，最终落在 T15 的某个 32 位寄存器里（课上现场算了两个例子，结论是 T19 之类的位置；规律可算即可）。

## 六、手写一次 mma：从 shared memory 到寄存器

例子：A 和 B 放在 shared memory 里，A 是 16×16（M×K，K 方向连续），B 是 8×16（N×K，K 方向连续）。

加载 A 的 fragment 时，需要根据 lane id 算出自己在 warp 内的编号，进而算出 GID/TID（在 8×8 小矩阵内的位置），然后按顺序把 4 个小块里我要的数全都拿回来，放到 4 个 uint（A0~A3）寄存器里；B 同理，拿 2 个寄存器。这些计算是在运行时做的。

避免 bank conflict 的办法：A 本来是 16×16，把它声明成 16×17（pad 一列），跨度改变后就不会有严重的 bank conflict 了；B 同理。SM80 上大家一般会用 pad。

然后发射 mma：声明 C 和 D 的寄存器组，把 A0~A3、B0、B1、C、D 填到指令对应位置，发射 tensor core 指令。8 个 cycle 之后，结果就在 D 的 fragment 里（D 是 16×8，也有自己的 fragment 布局）。最后根据 D 的 fragment 位置把它写回 shared/global memory。

这个过程很烦：需要手算 lane id、GID/TID。硬件为此提供了一个东西叫 **ldmatrix**（`ld.matrix.sync.aligned.m8n8.x4...`）：

- 它只关心 8×8 小块的起始地址和类型，从 shared memory 把数据搬进寄存器；x4 表示搬 4 次（给 4 个起始地址）；
- 地址语义比较反直觉：每个 lane 填的地址是某个 8×8 小块的**某一行的首地址**，不是 lane 期望拿到的元素地址；但它保证你最终拿到的 4 个数就是你要的 fragment；
- **它不帮你消除 bank conflict**：它只负责根据 shared memory 里的内容把数据摆成 fragment；地址落在哪些 bank 上仍然由 shared memory 布局决定。如果行跨距是 128B 的整数倍，就会出现超级严重的 bank conflict。举例：row stride 是 32B，正好是 8 个 bank 一组，需要发 4 次请求；row stride 是 128B 时，lane 0~15 和 lane 16~31 的 bank 完全相同，每次只用同样 8 个 bank，会有巨量 bank conflict。

解决办法就是 pad：不要把 row 长度做成 128B 的倍数，pad 到比如 132B，就不会有那么严重的 bank conflict。

### SM80 小结

SM80 的 tensor core 本质是 warp 级语义：一个 warp 里 32 个线程一起发起指令。数据流是：global memory → shared memory →（ldmatrix）→ 寄存器 → mma 指令 → D 留在对应线程的寄存器 → 写回 global memory。需要掌握 fragment、用 ldmatrix、注意用 pad 避免 bank conflict。

## 七、SM90（Hopper）：wgmma

SM90 的想法是继续扩大 MNK、进一步提高计算强度。具体做两件事：

1. **提高每次读取的复用率**：把 tile 增大；
2. **让操作数不要再经过寄存器**：A、B 直接放进 shared memory。因为寄存器是最贵的资源（读写灵活、带宽大、使用频率高），做不到翻倍（比如从 160 到 320 byte/cycle）。

同时，tensor core 计算时 CUDA core 是空闲的，所以希望让 tensor core 与 CUDA core 异步——tensor core 本质上不依赖任何线程操作，可以变成异步。

### wgmma 指令

SM90 上叫 **wgmma**（warpgroup MMA）：一条指令驱动整个 SM 的四组 tensor core。和上一代不同，上一代的 mma 是 warp 级的，每次只调用一个 sub-partition；wgmma 把四个 sub-partition 放到一块、一起发射这条指令。

`mma.sync.aligned.m64n64k16.row.col.f32.f16.f16.f32`：

- 它是一个异步的 MMA（发射时所有指令要同步、同时执行；tensor core 本身是异步的）；
- M64 N64 K16；
- A 是 64×16，B 是 16×64，都不再是寄存器，而是 **64 位 descriptor**，指向 shared memory；
- D 还在寄存器里：每个线程要准备 32 个寄存器；
- 有 scale 字段：为 1 时 D = A×B + D，为 0 时 D = A×B；还有整体取负、转置等小功能。

为什么 M 还是 64？每个 sub-partition 一个 warp、每个 warp 16 行，四个 warp 合起来 M=64。一个 warpgroup（WG）是 128 个线程。K 由数据类型决定：FP16 时 K=16，FP8 时 K=32（和 A100 类似）；N 从 8 起步、步长 8，最大可以到 256（M64N256K32 之类）。行列主序等信息全部移入 descriptor。

### wgmma 的三个限制

1. **只在 H100/H200（SM90a）上，没有兼容性**：编译时要加 arch SM90a；
2. **D 依然在寄存器**：M64N256 时每个线程要 128 个寄存器（线程寄存器上限约 255），压力非常大；
3. **异步带来两个问题**：什么时候能安全地读写累加器 D（结果是对的）；什么时候能安全地复用 A/B 所在的 shared memory。

### Proxy：异步与同步的可见性

wgmma 和 CUDA core 在两个地方执行。如果需要强同步（CUDA core 写完、tensor core 立刻看到），就要把两个分开的元件做强同步，代价很大。所以有了 proxy 的概念：

- **generic proxy**：所有 CUDA core 计算、普通 load/store 走 generic proxy。同一 proxy 内的操作是保序的——先写的操作，后读一定能看到；
- **async proxy**：TMA、wgmma、后面的 tcgen05 走 async proxy。为了让两边的可见性正确，发明了这两个 proxy 概念。

要让普通 store 写入 A、B 之后 wgmma 可见，需要用 `fence.proxy.async`（配合 async shared memory 和 barrier 之类操作）在中间拦一下做同步；反过来，async 写完要让 generic 看到，一般用 mbarrier 的 acquire 语义。

### wgmma 的生命周期

128 个线程组成 warpgroup：先 fence，发射异步计算，然后做 **commit group**，把前面一系列指令打包成一组，等待时只需要等这一组。发射之后 tensor core 继续算，CUDA core 可以继续做别的（比如对上一组的结果做后处理）。用 `wait_group N` 阻塞直到"小于等于 N 组在飞"：比如有 1、2、3、4 四组 mma，`wait 0` 需要四组全跑完；`wait 1` 只需等 1、2、3 组跑完，第 4 组还在跑时就可以操作前 3 组的结果。MMA 本身只是保序的（先发的先做）。

使用规则（五条）：

1. 发射到 wait 完成之间不能碰累加器 D（读写都不行）；
2. 未完成之前不能修改 wgmma 正在读的那块 shared memory；
3. A/B 如果走 generic proxy 写入，wgmma 默认看不见，要 fence；如果 A/B 是 TMA 进来的，TMA 是 async proxy 的异步拷贝，tensor core 能看见，不用插同步，只需要通过 mbarrier 保证它已经拷贝完；
4. 改过 D 之后，下一轮 wgmma 之前必须重新 fence，CUDA core 的修改才能被 wgmma 看到；
5. 违反的话编译器会给你 PTX warning，并强行插一个 `wait_group 0`（把所有组都等完才让 CUDA core 动），整个 pipeline 就全完蛋了。

commit 和 wait 之间的窗口可以用，也可以不用：可以做 app logic，也可以多个 warpgroup 分工（producer/consumer 结构）。最终目标是在异步 wgmma 的过程中同步地发射 CUDA core 指令，把 CUDA core 的代价隐藏好。另外 SM90 的 A 也可以通过寄存器读，主要是方便对 D 做了处理后把 D 再作为操作数。

### FP8 的 fragment（Hopper 第一次引入）

Hopper 上第一次引入 FP8，专门看一下它的 fragment：4 个 FP8 打包成一个 32 位寄存器；M 侧分 4 组（每组 16），每个 16 里分上下两半，每个 lane 出现两次；K=32 固定，每个 lane 读 4 个寄存器。D 的 N 可以以 8 的倍数额外扩大；16 行由 16 行一个 warp 负责。

## 八、descriptor 与 CORE matrix（core matrix）

descriptor 是干嘛的？它告诉你数据已经按"你能识别的格式"摆放好了，你只要照 descriptor 去读，就一定能取到你要的数据。

有一个概念叫 **core matrix**：

- 对任何元素，把 128 字节（16B×8）打包成一个单元。逻辑上它的基本元素不再是单个元素，而是 16 字节的一块；
- 例：64×16 的 A（M×K，FP16）：每行 8 个 FP16 = 16 字节，8 行构成一个 core matrix；整个 A 是 8（M 方向）× 2（K 方向）= 16 个 core matrix；
- core matrix 在 shared memory 里的排布有两种：**先摆 M 后摆 K**（每个 core matrix 之间相差 128B，每组之间相差 1024B）或**先摆 K 后摆 M**（组与组之间只差 128B，同组的下一个差 256B）。两种 layout 都可以被 tensor core 直接读取；
- tensor core 每次按 128B 的 core matrix 读取——正好 shared memory 每个 cycle 的吞吐是 128B。

descriptor 的内容：start address（元素在 shared memory 里的 offset，因为按 16B 对齐，所以右移 4 位）、leading dim（stride）、以及 swizzle 设置（防止 bank conflict 的两位）。tensor core 拿数据时需要知道你在 shared memory 里做了怎样的 swizzle 排布转换，才能知道怎么读，所以 descriptor 里要告诉它你做了哪些 swizzle、对 shared memory 元素做了什么操作。

## 九、Bank conflict 与 swizzle

回顾：shared memory 有 32 个 bank，每个 bank 4 字节；每次访问覆盖 32 个 bank 性能最佳。core matrix 以 16B 为单位（16B 对齐），4 个 bank 合成一个 **bank group**；一个 core matrix 有 8 个 16B 单位。

问题：如果按 128B 对齐分块，读一个 core matrix 会全部落在同一个 bank group 里——地址的低位中，core matrix 8 行的那些位在变动，但对 bank 有影响的位（bank group 那 3 位）在一个 core matrix 内是固定的，于是"撞死"。

解决办法就是 **swizzle**：对 core matrix 做变形，把每一行映射到不同的物理槽位，保证读一个 core matrix 时把 32 个 bank 全部用上。比如读 core matrix 0 时，把它的 8 行拆散映射到 8 个不同的物理槽位，就能直接用满 32 个 bank、消除 bank conflict；同时保证 8 个 core matrix 都有地方放。这个规律不用自己算，硬件/编译器会帮你算好。

## 十、手写 wgmma（naïve 版本）

不用 swizzle、直接手搓 wgmma 其实很简单，脚手架是：

1. 给 A、B 做 shared memory 里的描述符（直接摆放：A 是 M×K，B 是 K×N）；
2. 摆好数据后先 `__syncthreads()`，同步整个 shared memory 的读写；
3. 做一次 fence，让普通 store 写的 shared memory 对 async proxy 可见；
4. fence 整个 warpgroup；
5. 发射 wgmma（把对应的寄存器和 descriptor 放进去）；
6. `commit_group`；
7. `wait`。

这是最 naïve 的版本：它没有交错使用 tensor core 和 CUDA core。

### SM90 小结

SM90 的重点是：shared memory 直接喂进 tensor core，通过 core matrix 和 matrix descriptor 直接从 shared memory 取数据；D 仍然放寄存器。与 SM80 相比：发射者从一个 warp 变成四个 warp（warpgroup）；A、B 挪进了 shared memory；改成异步；shape 随之扩大。需要掌握 matrix descriptor；后面还有 TMA（会在之后的课程提到）。

## 十一、SM100（Blackwell）：tcgen05

SM100 的 tcgen05 主要做两件事：**把累加器挪进 TMEM（tensor memory）**，以及**不需要整个 warpgroup 发射、改成单线程发射**。

wgmma 遗留了两个问题：

- 累加器还在寄存器里：N=256 时要占 128 个寄存器，非常多；对 D 的操作会引发 wgmma 串行化，程序非常不好写；
- 发射一条 wgmma 仍然需要 warp 的 128 个线程一块执行，其实没必要，还加了同步成本。

于是有了 TMEM：

- 每个 SM 有 256KB 的 TMEM；它的组织形式是二维的：128 行 × 512 列，每格 32 bit；
- 存储方式比较特殊：高 16 位是 lane（行）、低 16 位是列，按 128 行重排序列组织；它和整个线程堆一样大，相当于 SM 上开了一块专门的大寄存器堆来处理累加器问题；
- TMEM **不能被 CUDA core 直接访问**（虽然看起来像寄存器），必须用 `tcgen05.ld`/`st`/`cp` 操作；
- 一个 warp 只能访问 lane 32I 到 32I+31（对应自己的 32 行），所以访问 TMEM 仍然需要 warpgroup 形式；
- 分配粒度是**列**：分配一列会给整个一列的所有行。M128 N256 时累加器刚好是 128×256（半块 TMEM），特别适合 double buffer 的 case——不用等累加器搬走就能接着算；
- TMEM 分配要手动：让你搞一个 shared memory 地址，分配结果放在 shared memory 里（不是寄存器）；分配的列数必须是 2 的幂且不小于某个下限（字幕为"2/3"，听辨不清）；分配由一个 warp 跑，但结果要 4 个 warp 一起用，所以塞到 shared memory；
- 分配完之后要声明"不再分配了"，否则其他 CTA 会卡住等释放；用完必须释放（single thread 做 deallocate）。

### tcgen05.mma 指令

`tcgen05.mma.cta_group::1...`：

- 不再需要 sync、aligned——**一个线程就能发射**；
- 所有第五代 tensor core 指令以 tcgen05 开头；cta_group::1 表示一个 CTA 跑（后面还有两个 CTA 的情况，稍后讨论）；
- 指令里给出 tensor core 类型、D 和 A 的地址、B 的 descriptor（从 shared memory 取）；
- 用一个 **idesc**（instruction descriptor）描述 tensor core 操作：D 类型、A 类型、B 类型、AB 取负、AB 转置、M、N、K 等，一个 32 位就能放下——因为类型组合特别多，用一个 descriptor 描述可以避免引入太多指令类型；
- 支持 F16、TF32、I8 等类型；还有 M×P8 乘 M×P4 的组合（DeepSeek 那两个 V 里的 GEMM 就是这种），以及纯 P4 的版本（K 大一倍、快一倍）。

### mbarrier

怎么知道 tcgen05 是否完成？wgmma 有 `wait_group`，但 tcgen05 用的是 **mbarrier**——这是本来就有、可以拿来做同步的东西（cp.async、TMA 也可以用它）。它是什么：

- 位于 shared memory 里的 64 位对象，有几个字段：**phase**（0/1 单调递增；达到条件才 +1、基础出现反转，这样你就知道"我要等的东西完成了"）、**arrive count**（cp.async 用的：多少个线程一块发 cp.async、要到达多少次）、**ts count**（TMA 用的：要传多少字节、等到多少字节到达才做反转）。在 arrive count 和 ts count 同时为零的情况下，phase 加一。
- 谁来 arrive：线程自己可以 arrive（手动触发加一）；TMA 搬运完成后会记录 ts count；`tcgen05.commit` 后、发射过的 tcgen05 操作全部完成时也会有一次 arrive。
- 谁能等：所有线程都能 `try_wait`（只要能访问到 shared memory 就能等）；而 wgmma 的 wait group 只能发射者自己等。
- 等待的语义：init（初始化）、try_wait（尝试等待；硬件可以挂起等待的线程去跑别的工作）、parity（只关注相位翻没翻转，不关注具体值）、acquire（等待之前的内存操作在等待之后可见，相当于同步）、relax（只等待触发，不做内存保序）；还有 `cta::shared`（barrier 在当前线程块内成立）和 `cluster`（可以等别的 CTA 的 barrier）。

`tcgen05.commit` 和 wgmma 的 commit group 逻辑相似：一个线程发 mma、再发 commit，把 commit 挪到 mbarrier 里；warpgroup 里的 4 个 warp 就可以等这个 mbarrier 被唤醒——tcgen05 完成时会在对应的 mbarrier 上做一次 arrive。

### 一个完整的 pipeline 骨架

1. 先 TMA 把数据搬到当前的 shared memory buffer；
2. 一个线程连发 block 分块数量那么多的 mma；
3. `tcgen05.commit` 把所有这些 mma commit 到一个 mbarrier；
4. 需要读 TMEM 的线程等 mbarrier 相位翻转，然后开始从 TMEM 取数据；
5. `tcgen05.ld` 取数：sync.aligned 又回来了——搬数据需要每个线程各自参与；先算好从哪里开始搬，一次搬 32 行里的 8 个 32B 元素，搬 8 次，结果放 8 个寄存器（具体 shape 直接看 PTX 文档）；
6. 同步规则：tcgen05 在 async proxy 上，shared memory 里用普通 store 写入的东西需要 `fence.proxy.async.shared::cta` 才能被它看到；跨线程场景下（线程 A fence、线程 B 做 epilogue 之后线程 A 再跑 tcgen05），要在 sync arrive 之后、跑 tcgen05 之前声明"我开始跑了"，不要把访问它的操作排到它前面；
7. `tcgen05.cp`：从 shared memory 拷贝到 TMEM 的异步拷贝，主要用途是把 scale factor 搬进 TMEM；cp 与 mma 是保序的，cp 之后不用同步再发 mma（mma 会等 cp 跑完）；
8. 七步流程：分配 → 把 A、B 写到 shared memory → fence 让 tcgen05 能看到 → 发射 → 等完成 → 读累加器 → 释放。

单线程发射的实现：用 `elect.sync` 从参与指令的所有线程里选出一个，拿到 predicate；被选中的线程做 tcgen05 fence、发射 tcgen05、commit 到 mbarrier；每个 warp 读自己对应的 32 行，读完一个 warp 释放 TMEM，然后做后处理。

## 十二、Two-CTA 的 mma

观察：如果做 A×B，把 M 扩一倍，N、K 的矩阵其实可以复用。所以两个 CTA 一起发射 GEMM 时，每个 CTA 只要放一半的 B，从 shared memory 里读 B 就可以了：

- 两个 CTA 一起发射（cta_group::2）；偶数 CTA 是 leader，存 A 的上半，另一个存 A 的下半；
- B 共享，两个 CTA 各存一半 B，tensor core 从 shared memory 搬 B——shared memory 压力小一点，整体计算强度也提高了；
- M 最大可以达到 256；
- 通知完成：把 mbarrier 改成 multicast，cluster 里就可以用这个 barrier 等——比 SM90 里 wait group 之后还要用别的方法通知一个 warp 更好；
- 两个 CTA 必须落在**不同的（相邻的）SM**：launch 时把两个 CTA 作为一个 cluster，就能保证它们被调度到相邻 SM 上；
- 这是硬件设计：两个 CTA 同步、一起做矩阵，中间有硬件通路（一个 SM 的 shared memory 与另一个 CTA 的 tensor core 之间有专门的线，不走通用的跨 CTA 通信）；
- 好处：shared memory 里存的东西减半、读取压力减少，同时 M 翻倍——本质就是更好地复用 B。

## 十三、三代 tensor core 的对比

三代 tensor core 越来越"轻"，越来越像专用硬件、从 CUDA core 剥离：

- SM80：与 CUDA core 结合很紧，A、B、C、D 全在寄存器，靠 warp 发射；
- SM90：有一点梳理——异步、A/B 进 shared memory、warpgroup 发射；
- SM100：非常像专用的硬件了，和 CUDA core 没什么关系，连读 TMEM 都要通过特殊操作。

## 十四、低精度与 block scaling

### 为什么需要低精度

每代算力都在翻倍，但还是不够。目标：减少显存与带宽占用、提高计算强度、探索更低的位数。Hopper 是第一代支持 FP8 的 GPU。FP8 对精度的保证很低——它的最大值只有 448，不能保证所有数都能表示出来，但计算量加得特别多。

关键概念是 **scale factor**。

### FP8 格式

FP8 的 E4M3：最大值 448，最小正正规数 2⁻⁶，最小次正规数 2⁻⁹；4 bit 指数 + 3 bit 尾数；因为 FP8 没有 NaN 占用一个编码，所以比 half 标准多出一档可用数值。但它的动态范围和精度仍然非常难看。E5M2 更激进：为了更大的动态范围，尾数只有 2 bit，相对精度误差更大。

FP8 的动态范围肯定不够表示训练中一个 tensor 的分布，所以要配 scale factor。

### Per-tensor scale 的问题

一种方案是 per-tensor scale（英伟达在 Hopper 上一直想做的）：直接取 tensor 里的最大值，把所有值除以它、取最近的 FP8。问题：大部分值在 0~1 之间时还能做，但如果突然出现一个 3000 的 outlier，1.0、0.5、0.1 勉强还能看出来，再小一点的数字就直接变成 0 了——这一块信息就全丢了。所以这样不行。

### Block scaling（DeepSeek 的做法）

可行的方案是 **block scaling**（per-block scale），有两个要求：

1. scale 不能随 MK 变化——K 在变化的时候最好不变，否则 tensor core 就用不了了；所以 scale 沿 K 需要是分段常数，每一段内部的 scale 是常数，可以提到 GEMM 外面，GEMM 做完之后把 scale 乘上去（结果是 FP32，可以用 FP32 乘）。

这就是 DeepSeek V3 的 recipe：激活按 token 切（每个 M 是一个 token），K 维在 token 内每 512 bit 分成 4 格、每格 128B——这刚好是它对 K 切块的粒度；B（K×N）的粒度没那么细，按 K 和 N 都有 scale。算 GEMM 的一块时，乘以这一块两边的 scale。

### H100 累加器精度问题

这里有一个 NVIDIA 从来没有提过的问题：**H100 上 FP8 GEMM 的加法累加只有 14 bit**。FP32 的尾数是 24 bit，但实际参与累加的只有 14 bit，会丢掉 10 bit。K=496 时相对误差达到 2%。如果完全用 tensor core 累加，一组尾数乘积按最大指数右移对齐、保留 14 bit 相加，其余 10 bit 全丢了——误差比较大，很难接受。

DeepSeek 用的方法是 **promotion 循环**：算一段、累加一段，用 FP32 全精度累加。它有两个累加器：一个是 CUDA core 拥有的真 FP32，另一个是交给 wgmma 的临时累加器。每 128 个 K 做一次 promotion、做一次正儿八经的累加，把精度留住。代价是 CUDA core 占用：DeepSeek V3 时 23% 的时间花在 promotion 上。

把同样的思路扩展到 B200 是不行的：B200 上 tensor core 和带宽都变更快了，但 CUDA core 没有快很多，会有 44% 的时间花在 promotion 上——不可接受。

### B200 的硬件 scaling

所以 B200 这一代芯片有了**硬件的 scaling**，主动支持一些低精度格式，比如 FP6 和 FP4：E2M1、E3M2、E8M0 等（存储格式），以及 E4M3。存储格式与打包格式不同：存的时候按 FP 存，但会有 scale。MX 格式每 32 个 K 一个 scale；N 上配置的话每 16 个 K 一个 scale。scale 类型也不一样：MX FP8 的 scale 是 E8M0（只有指数部分，左移/右移）；OFP4 的 scale 是 1U8（听辨存疑，可能对应某种 8 位无符号格式），可以更精确地控制 scale。

MX 是 OCP 做的标准，AMD、ARM、英特尔、Meta、微软和 NVIDIA 一起发布的，大家都支持，现在包括华为的卡也支持。用 OFP4 有更好的精度，但可前进性（兼容性）不太好。

scale factor 在 TMEM 里也要放得下。有个特殊点：TMEM 必须复制在四个分区——scale factor 的四个分量要分别在四个 lane 里各存一份相同的值，这样运行起来更方便。

### 存储量化 vs 计算量化

说"量化"时其实有两种：

- **存储量化**：比如 W4A16——显存里存 int4，但计算时 dequant 成 BF16、放到寄存器里用 BF16 算。它只减少显存需求（显存占用），没有加速计算的作用；
- **计算量化**：今天讲的 FP8/FP4 计算量化——存 FP8、算 FP8，最后累加是 FP32。操作数真的以低精度进 MMA，算力会成倍翻。

量化的力度有几个维度：

- 量化什么：权重 W、激活、KV cache、累加器、梯度、优化器；
- 怎么量化：per-tensor 不行；per-block、分组量化、按 KV head 量化；
- scale 用什么存；
- 什么时候量化：训练后量化（PTQ）、训练时量化（QAT）、还是训练本身就跑在低精度上。

比如 DeepSeek：训练应该是在 FP8 上跑的（也可能 FP16），最终给的结果是 MX FP4 权重 × MX FP8 激活的流式推理；这需要一系列后训练，把"毛刺"消掉，让它对量化结构友好——既降显存、提高计算性能，又尽量少影响精度。

## 十五、总结

今天我们讲了三代 tensor core，核心问题是：**算得方便之后，如何把数据喂好**。

- **SM80**：A、B、C、D 全在寄存器，每个线程自己算地址取数，需要掌握 fragment；
- **SM90**：累加计算器不够、tensor core 计算时 CUDA core 空闲，所以 A/B 直接放 shared memory、用异步方法做 GEMM（wgmma），配合 matrix descriptor；
- **SM100（B 系列）**：累加器占大量寄存器，而且希望硬件有 scaling，所以有 TMEM、单线程发射、mbarrier、two-CTA，以及硬件 scale。

之后我们还会讲 tiling、pipeline 以及 warp specialization，用来补上"计算强度还远远达不到要求"这个问题。

## 十六、问答环节

- **单线程发射有什么优势**：一是可以不用同步；二是可以给这个 warp 分配很少的资源。
- **Two-CTA 是硬件层面还是软件层面的设计**：是硬件设计。tcgen05 的 cta_group 是两个 CTA 一起做同步、一起做矩阵，SM 的 shared memory 与另一个 CTA 的 tensor core 之间有专门的硬件通路，不走通用的跨 CTA 通信。好处是 shared memory 里放的东西减半、读取压力减少、M 翻倍。
- **两个 CTA 必须位于不同的 SM 吗**：是的，必须是相邻的 SM；launch 时把两个 CTA 作为一个 cluster，就能保证它们被调度到相邻的 SM 上。
- **单线程发射会不会浪费线程**：warp 如果啥也不干，不会被调度上去，不占资源；在体系结构设计层面未来可能有演进，但在当前设计上没有"浪费"的说法。
- **M=256 的 two-CTA 与 M=128 对比**：本质是复用 B——B 从 shared memory 拿，两个 CTA 各存一半 B，shared memory 容量和读写压力变小，同时 GEMM 本身（M）扩大，所以统计上更快。

---

## 附：听辨存疑与说明

1. 主讲人姓名：字幕音译"孙晓航"，B 站简介为"孙远航"，正文按简介采用，未能完全确认。
2. TMEM 分配"列数必须是 2 的幂且不小于 2/3"（字幕）：下限数值听辨不清，未强行补全。
3. OFP4 的 scale 格式"1U8（UECM3）"：听辨不清，未强行补全。
4. DeepSeek 相关细节（"两个 V"、训练精度 FP8/FP16）按字幕保留，未补充外部信息。
5. 数值（0.25、9.75、19.5、3.2、26 倍、311.9 T、156、8 cycle、1.41 GHz、128 个寄存器、256KB TMEM、448、2⁻⁶/2⁻⁹、14 bit、2%、496、23%、44%、32 个 K、16 个 K）均按字幕原样保留。
