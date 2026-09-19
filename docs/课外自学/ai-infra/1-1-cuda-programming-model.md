# 1.1 CUDA Programming Model —— 完整讲课转录

## 视频信息

- 标题：北京大学未名超算队 × LCPU AI Infra Seminars 系列讲座：1.1-Cuda Programming Model
- 链接：https://www.bilibili.com/video/BV1g83w6DETm
- UP 主：北京大学Linux俱乐部
- 主讲人：郑熠（北京大学信息科学技术学院）
- 时长：1:07:22
- 作业仓库：https://github.com/lcpu-club/wmhpc-training-camp-x-lcpu-ai-infra-seminars

## 校订说明

本视频在 B 站没有 AI 字幕，本转录基于通义听悟导出的逐字稿（`1.1-Cuda Programming Model_原文.srt`）校订整理。视频为中文讲授，正文沿用中文。原始逐字稿覆盖全片（末条时间戳 01:07:21.590，视频时长 01:07:22）；其中个别片段含 ASR 乱码，已按上下文恢复语义，无法确认处已在正文标注。

---

## 开场

我给大家简单介绍一下 CUDA 编程模型。首先第一个问题：**我们为什么要用 GPU**？

大家都知道 GPU 比 CPU 快很多，那么这个"快"是哪方面的快？看这张图：一个显著特征是 GPU 把大量的晶体管分配到了计算单元上，而 CPU 把大量的晶体管分配到了控制单元上。CPU 做了大量的控制，但它的不足是计算单元少；GPU 相反，计算单元多是它的优势。

注意：我说 GPU 控制少，不是说写 GPU 代码时可以少做控制——实际写代码的体验跟这句话是相反的。CPU 因为硬件自动做了大量控制，我们写 CPU 代码时只需要做少量控制、把注意力集中在计算上；GPU 因为硬件控制做得少，编程时需要把大量注意力集中在控制上。所以 CPU 编程还是要更方便一些的。如果说 CPU 是我们这一代程序员的摇篮，那我们肯定不能一直待在摇篮里。

## 一、吞吐量 vs 延迟：GPU 为什么快

GPU 是为了**吞吐量**而生的。从硬件厂商的角度说，要提高一个东西的吞吐量，只需要堆叠硬件就可以了——两台电脑的吞吐量就是比一台大。但要减少延迟的话，要考虑的事情就很多。

以加法为例：CPU 只需要一个时钟周期的延迟就可以完成最简单的运算，GPU 却要用四个时钟周期。GPU 解决延迟劣势的方法是**隐藏延迟**：它同时接收非常多可以并行处理的任务，让计算单元在等待数据的时候总有活可以干。"同时处理足够多的任务"是 GPU 的硬件创新——如果这是一个软件实现，GPU 的编程量会变得非常恐怖，因为做加法都需要排流水线，不能立刻算出来。**GPU 擅长隐藏延迟**。

另外 GPU 不仅计算吞吐量大，缓存吞吐量也比较大：一条 GPU 用的主存 HBM，存储量可以达到 DDR5 的 20 倍（这个说法有一点夸张，因为插 DDR5 时一个机器上可能插 20 条甚至更多也没问题，而 HBM 的堆叠要稍微困难一些）。要达到 HBM 标称的访问速度，必须做地址连续的大块访问，随机访问是跑不满的——内存带宽大是这个意思。

### N² 过百万的讨论

前些年有个说法叫"**N² 过百万**"：一个运算量有 100 万平方的程序，大概几秒钟就能跑完。我们现在有没有做到？严格来说没有。看三个递推式：

1. 第一个递推式不是 O(N²) 递推的，为公平起见我们要求它的第 10¹² 项。它是个强制在线问题，只有算出了前一项才能递推出后一项——需要一个极高频率的计算设备才能算出来，而 GPU 显然没有（事实上 GPU 的计算频率比 CPU 还低一点），所以它很困难；
2. 第二个是组合数的递推，本身非常适合并行计算，但 GPU 露出了怀疑的表情：它有潜在的 memory-bound 风险。比如把杨辉三角前 100 万行、100 万列打印出来，瓶颈就不再是计算，而是把算出来的结果写到某个存储器里——随便哪个存储器都写不动；
3. 第三个计算式比较适合 GPU 并行。我没有说它一定不能在 O(N²) 以下复杂度内算完，但至少朴素的 O(N²) 递推对 GPU 非常友好，适当编写程序可以让 GPU 算力更充分地发挥。它有一个潜在问题是使用了除法——GPU 算除法的吞吐量比标称计算量慢，大概是 1/8 的样子，除法的开销还是大的；但代码写得好一点，上 N² 过百万没有大问题。

（评论区问为什么是 O(N²）：要求它的第 N 项确实要 O(N²)。）

## 二、CUDA 的五个核心概念

CUDA 是英伟达自主研发的编程模型。里面列出的五个概念至少有四个是英伟达的原创——它们的实际意义不大，如果换一家显卡厂商编程，可能需要熟悉另一套术语体系。五个核心概念：

- **Thread**：程序运行的基本单元；
- **Warp**：32 个 thread，这 32 个 thread 需要执行相同的指令；
- **Block（又叫 CTA）**：线程间合作的范围；
- **SM**：执行 block 或者说 warp 的硬件。

看一个比较旧的 SM 示意图：SM 里面有一个 warp execution context，标了 warp 0 到 warp 63——也就是说这个 SM 可以塞进去 64 个 warp。但"塞进去"不等于"一起跑"，实际能同时跑的可能只有四个。SM 里还有一些细节结构（cache、warp selector 等）先不说，以后细讲。

- **Grid**：block 的集合。

从右边的绿色图可以看出，block 的数量可以远多于 SM 的数量。那怎么执行？只能排队。但注意：**排队的顺序（block 的调度顺序）是未定义的**——可能我心里想好的 block 分配顺序跟它实际执行的顺序不太一样。

左边这张图是软件跟硬件的对应关系：thread、block、grid 是偏软件的概念；右侧的 CUDA core、SM、GPU 是实际的硬件实现（这张图经过了艺术加工）。一个 thread 不能单独跑，从显卡上掰一个 core 下来也不能跑——GPU 的 core 不像 CPU 的 core 那么独立。

上一页说了 block 的调度顺序是未定义的，所以设计 block 划分时，心里要想：**每一个 block 都是可以独立执行的程序，block 之间不可以有数据依赖**。这带来一个好处：软件跟硬件有了一定程度的解耦——我们不需要关心 GPU 上到底有多少个 SM，有一个 SM 这个 kernel 能跑，有一千个 SM 也能跑。右边是个例子：假如程序包含 8 个 block（8 个可以独立计算的部分），在有 2 个 SM 的 GPU 上，可以把 4 个 block 一组分给两个 SM；如果有 4 个 SM，就两个 block 一组送一个 SM。这给了 CUDA 代码更好的硬件兼容性。

## 三、CUDA 的编译原理与执行流程

在讲真正的 CUDA 代码之前，非常简略地讲一下编译原理。CUDA 扩展了 C 和 C++ 的语法，所以 CUDA 代码需要有自己的后缀名和自己的编译器（nvcc）。一段 CUDA 代码包含在 GPU 上运行的部分和在 CPU 上运行的部分，这两部分由不同的编译器编译，编译好之后塞到同一个可执行文件里（可执行文件是 CPU 上才有的概念）。CUDA binary 是 GPU 的可执行文件，相当于比较低级的汇编/机器码；GPU 代码编译的中间产物是所谓的高级汇编 **PTX**，它的兼容性要比 CUDA binary 好一些。

编译出来的 CUDA 程序最终的工作流程：

1. 程序在 CPU 上被调用，在 CPU 上准备好数据；
2. 把数据从 CPU 的内存拷贝到 GPU 的内存（cudaMemcpy）；
3. 通过一个叫 **kernel launch** 的特殊函数把它送到 GPU 上并行执行；
4. 把 GPU 上算好的结果拷回 CPU。

## 四、Kernel 设计实践

**Kernel** 就是在 GPU 上运行的函数（不严谨的说法）。为了写 kernel，先介绍几个关键字（nvcc 的扩展语法）：

- `__host__` 指 CPU，`__device__` 指 GPU；
- kernel 用 `__global__` 修饰，返回类型必须是 `void`；这个函数可以被任何地方调用，我们一般就是在 CPU 上调用、在 GPU 上执行。

### 最简单的并行程序：向量加法

一个非常简单的并行程序可以产生 N 个线程做向量加法（程序不完整，比如没给出 A、B、C 的定义）。这里有一个三个尖括号的语法，先当后面的 N 就是线程数。

向量加法 kernel 里的指针 A、B、C 不能用常规方法申请，CUDA 有自己的 API：`h_a` 指向 CPU 中的内存块，`d_a` 是 GPU 显存上的内存块，申请方式差不多——都要写 `malloc` 或 `cudaMalloc` 申请，最后 `free` 掉。CPU 与 GPU 之间的数据通信通过 `cudaMemcpy` 决定，它跟 C 的 memcpy 的区别是有第四个参数：host-to-device（从 CPU 粘到 GPU）、device-to-host（反之）。

刚才展示的 kernel 启动参数出现在 kernel 外面，这是原版 C 没有的语法，先介绍前两个参数：

- **gridDim**：这个 kernel 的 block 数量；
- **blockDim**：每个 block 里 thread 的数量。

所以这实际上是一个两层的分块结构。分块不限于数列上的分块，可以是 1 到 3 维的（毕竟显卡本来是做显示的，不是预给我们做数据结构提供的）。kernel 里有内置变量：`threadIdx.x`（kernel 里的内置变量，用于区分我是第几个被执行的线程）、`blockIdx.x`、`blockDim`、`gridDim`（发射时选定的两个参数）。

### 二维 block 的例子：矩阵加法

现在有一个 N×N 的矩阵 A 和一个 N×N 的矩阵 B，要逐元素相加放到 C。如果你深究实践细节，会想 CUDA 申请来的内存是不是一维的？这不重要——正常创建二维数组的方式就是创建一个一维数组，用行主序或列主序填充数据，并不会真的造一个二维数组出来。

稍微复杂一点的例子：使用多个 block，每个 block 限定为 16×16（256 个线程）。定位下标时，`threadIdx` 是你在块内的编号，`blockIdx` 是你在所有块的序列里的编号，由此算出实际要计算的 I、J。`if` 那一行是边界处理：因为矩阵大小经常不是块长的整数倍，用 if 判一下，不会有太多性能损失。

## 五、Warp 的细节

刚才 warp 基本一笔带过，只说了一个 warp 是 32 个 thread，这 32 个 thread 必须执行相同的指令。如果出现了分支怎么办？**分支的路径会串行化**：一个 warp 里 32 个 thread 做判断，比如 `if (threadIdx.x % 2 == 0)`——偶数 lane 走 A 路径、奇数 lane 走 B 路径，程序就出现了分支；分支的路径串行化，走当前路径的线程被激活，未走当前路径的线程被 mask 掉；if-else 块结束之后，线程还会合回来。

**thread 与 warp 的映射关系是确定的**：在同一个 block 里，thread 有一个严格的线性序号（thread id），由 threadIdx 计算出来。比如只按一维分块，id 直接等于 `threadIdx.x`，后面的维度都是 0；然后每 32 个连续的 id 给一个 warp。

## 六、为什么说 NVIDIA GPU 是 SIMT 执行方式

我要试图说服大家：英伟达的 GPU 是 **SIMT**（单指令多线程）的执行方式。主要原因：**数据是被动的，线程是主动的**。有人说 warp 对分支的处理是 SIMD 和 SIMT 的本质区别，我不这么认为——mask 这个东西在 SIMD 上也是有的。我认为 thread 和 data 的本质区别是：**thread 有自己的 program counter（PC，指向要执行的指令的地址），而 data 没有**——数据是被动被计算的。

一条 SIMD 指令肯定只有一个 program counter。从 CC（compute capability，可以理解成 CUDA 的版本号）7.0 起，每一个 thread 有自己的 program counter，那我刚才说的这段话就成立了。但是，虽然每个 thread 可以执行不同的指令了，执行不同的指令也不是一个推荐的操作——因为这些 worker 的 program counter 取值太多，SM 的指令调度器会非常低效，它每个周期只能让 warp 里面 program counter 相同的部分去执行指令。所以**最有利于 SIMT 发挥效率的编程方式仍然是 SIMD 式的**。

当然这个事情不绝对，见仁见智：我也看到不少资料直接说 GPU 是 SIMD，特别是说 warp 是 SIMD。怎么把两种观点统一起来？可以搬出冯·诺依曼的存储程序理论：只要说程序跟数据是一回事，那么 data 跟 thread 就也是一回事，问题就"杀死"了。虽然现在硬件实现上 program counter 还不是普通的寄存器（x86 的 RIP 跟别的寄存器还是不太一样），但也差不多。

## 七、Warp divergence 与 Better kernel

刚才提到的"同一个 warp 进入分支、不同 lane 走不同指令"就叫 **warp divergence**。Bad kernel 展示了一个 warp divergence：按奇偶分组，偶数下标做 exp、奇数下标做 sqrt——这很低效，因为每个 warp 相当于都得既走 if 又走 else。

**Better kernel** 是更好的写法：既然要奇偶分开，就先给偶数编一个号、再给奇数编一个号，这样所有涉及 exp 指令的编号连续了，所有涉及 sqrt 指令的编号也连续了（除了中间有一个无可避免的分段点）。于是在 Better kernel 启动后产生的那么多 warp 里，只有最多一个 warp 会既走 if 又走 else，其余 warp 都只进入其中一个分支——这是相当高效的执行方式。

想想 CPU 会怎么做：奇偶分开其实是一种可以被分支预测非常好地掩盖的执行模式。CPU 会发现"奇数位置我走了 sqrt 那条路、偶数位置我走了 exp 那条路，这两条语句在不断地交替"，于是被分支预测命中。而 GPU 没有分支预测这个功能，所以作为程序员，我们要手动给 GPU 赋予分支预测的能力——这就是显式编码分支预测的 Better kernel。

## 八、Warp level primitives：shuffle

warp 是一个比较偏硬件的概念，写比较基础的 kernel 时可以无视 warp 的存在，只需要关心 grid、block 和 thread；但把 warp 用好，计算效率更高。这里举一个简单的例子：用 warp 做 reduce，用到 `__shfl_down_sync`。

`__shfl_down_sync` 的语义：

- 第一个参数 full mask：有哪些 lane 要参与这个函数的调用，`0xffffffff` 说明这 32 个 lane 全部执行；
- 第二个参数 value：我们打算共享的那个变量；
- 第三个参数 offset：lane 下标之间的偏差；lane id 就是之前说的从 thread id 对 32 取余的余数。

右下角的图展示了 reduce 的一个局部：执行 `__shfl_down_sync(mask, value, 4)` 时，lane 5 会从 lane 9 那里得到值、相加得到 lane 5 的 value + lane 9 的 value；而 lane 28 试图从 lane 32 那里获取值，越界了，此时它的值是未定义的。所以做完 log 轮之后，只有 lane 0 的值还有定义，并且这个定义就是之前 32 个 lane 的和（block 层面实现 reduce 大概要用 shared memory）。

`__shfl_down` 可以说是一种 message passing。warp 内信息传递的方式非常丰富：有 shuffle down、shuffle（箭头从左上往右下指）、还有 `__shfl_xor_sync`——把下标异或上某个固定值。如果大家写过 FFT、FWT 之类的东西，会意识到在 warp 里做这些分治操作非常高效。这段代码跟最终执行的代码还有很大区别（还没展示 kernel 怎么调用它），这将是我们课程的一个作业。

## 九、Block 层次：shared memory、同步与 cluster

Block 内的线程比较类似于 CPU 的线程，所以它们也有 shared memory——区别是这个 shared memory 需要你手动声明（`__shared__`）。CPU 并行计算的常见操作——原子操作、内存屏障、线程同步——在 GPU 上都实现了一遍。这里只讲最简单的指令：线程同步 `__syncthreads()`。

**这个线程同步有一个坑**。对比上下两个 kernel：下面的 kernel 写得比较奇怪，含义是输入一个 128 位的 int 数组然后 reverse 一下，用 shared memory 有点含义，但先不管；要说的是 **`__syncthreads` 不能乱用**——它只能出现在 block 里所有线程都能运行到的分支里面。

- 没有分支最好，显然所有线程都能运行到；
- 如果有分支，要保证这个分支在块内是一致的，比如 `blockIdx`——它是 block 的属性，在一个块内一致；
- 下面那个 invalid behavior 的代码表面上看极其愚蠢：它的 for 循环让每一个线程都能恰好遇到一次 `__syncthreads`（`threadIdx.x` 的取值恰好是 0 到 `blockDim.x - 1`，这个 for 循环在所有线程里肯定都能跑到一遍），正常人类写不出这种代码。但如果我们写了，它是错的：因为你调用 `__syncthreads` 的时机跟别的线程不一样，这会导致未定义行为。

**Cluster** 是从 CC 9.0 开始支持的概念（这里简单讲，没有代码实现）：它不仅想在 block 范围内交流，还想在不同 block 之间交流。用 cluster 就可以：这些 block 被保证在同一个 **GPC** 上调度——GPC 是跨越 SM 的概念，一个 GPC 里可以包含若干个 SM，这样它们可以实现跨 block 通信。cluster 的引进也带来了一种新的内存，叫 **distributed shared memory**（不会细讲，进入内存层次结构时整体过一下）。

### 内存概念一览

- 每个 thread 可以有内存（一般更多的是寄存器）；
- thread block 层面可以申请 shared memory，方便线程通信；
- cluster 层面可以有分布式 shared memory；
- 整个 grid 都可以访问的是 GPU 的显存（global memory）。

这些是软件概念，硬件上的 L1、L2 并不在里面。需要注意的一点：**local memory** 的 location 在 device 上，但不一定在 SM 上——local 只保证 thread 的私有性，并不保证速度；一般来说能用寄存器的时候，编译器不会给你塞一个 local memory 进来。还有 **constant memory**：它有缓存，如果真的有常数，放进去会快一些（或者放 kernel 要传给 kernel 的参数）。

## 十、Shared memory 与一个矩阵乘法例子

正式介绍一下 shared memory：它和 L1 缓存是**同一块存储器的不同划分**。L1 缓存是硬件自动管理的，shared memory 是需要手动编程控制的——有一个可以手动控制、跟 L1 缓存一样快的东西，自然是极好的。

展示一下 shared memory 的威力（右边是一个矩阵乘法的代码；要写出矩阵乘法仅知道 shared memory 不够，这里只是鉴赏）。注意几个点：

- `restrict` 修饰符：这个修饰符 GCC 就有，CPU 上也能用；它的意思是告诉编译器"通过这个指针访问的元素跟别的指针没有关系，指针之间不会重叠"。读 C 源码的话，性能好的 C 函数（比如 memcpy）应该也有 restrict；也就是说，memcpy 时两段内存重叠是未定义行为。这里也一样，如果你给标了 restrict 的数组实际上是同一个数组的不同区间，那也是未定义行为；
- `__shared__` 修饰词说明它是 shared memory；
- `__syncthreads` 是之前讲的。

看这个代码在干嘛：

- 最后一行：矩阵乘法的结果只有一个赋值语句——这个 kernel 里**一个线程只负责计算矩阵乘法的一个元素的最终结果**；
- 没展示 blockDim 和 gridDim，但从最后一行可以看出：blockDim 应该是 tile size，一个 block 里有 32×32 个线程；gridDim 应该由输出大小 N×M 除以 32 分别决定（这里假设 M、N、K 都是 tile size 的倍数，少写边界条件）；
- 执行流程：每个线程分别加载一个 A 矩阵的元素和一个 B 矩阵的元素；32×32 个线程同时加载，得到两个 32×32 的子矩阵；在所有线程都把子矩阵的信息取出来之后，再做朴素的乘法；
- 一个细节：**X、Y 必须是反着写的**——对 A tile 的访问是先 Y 后 X，对 A 的访问也是先 Y 后 X，D 也一样。这应该是一个行主序：A 的前面（K× 的东西）是行号，后面是列号。可以发现 `threadIdx.x` 几乎总是被放在下标增加的最低位。

这背后的道理是 **memory coalescing（合并访存）**。写过 SIMD 的话这都不用解释——SIMD 最喜欢做的就是连续内存的 load 和 store。一个 block 被分成 warp 执行，根据之前 thread id 与 warp 的对应关系，同一个 warp 里 `threadIdx.x` 恰好取遍 0 到 31、而 `threadIdx.y` 是同一个值；访问二维数组时高维固定、低维是 0 到 31 的连续访问，这在 warp 层面是极其连续的访问，对内存的访问效率比较高。coalescing 有点像 CPU 的缓存行机制：访问一个地址的内容，不仅能得到这个地址的内容，它旁边连续的 32 字节地址也会被取回来，所以连续访问非常重要。

## 十一、GPU 的硬件实现

这是一个非常概括的图：一个 GPU 首先连着一个 DRAM（大概几百 G 的 HBM，非常快、带宽非常大）；最接近它的缓存是 L2 cache（没有 CPU 那样的 L3 cache）；再下面是我们之前提过的 GPC 和很多很多的 SM。

放大一个 SM 看一眼（Blackwell Ultra 的 SM）：

- 一个 SM 里面有大概四个分区，这四个分区都可以调度 warp，所以一个时钟周期一般可以给四个 warp 发射指令；
- shared memory 和 L1 缓存合起来大概是 256KB，可以切一些范围出来；
- 在四分之一 SM 里面有一个非常耀眼的 tensor core（这次不会讲）；
- thread 的执行单元是 CUDA core（主要是 CUDA），当然还可能有 SFU（special function unit）来执行一部分指令。

### CUDA core 与 SFU

简单的加减乘用 CUDA core 算肯定没问题，而且吞吐量特别大：比如一个 SM 算加法，一个周期可以算 128 个加法（因为它有 128 个 core）。而如果想算除法，必须先经过 SFU 算一个倒数，吞吐量就下来了——大概每个 SM 一个周期只能算 16 个数的倒数，而且必须是 32 位的 float，64 位更慢，并且这个倒数是**近似**的。

这里放了平方根倒数，让大家感受一下近似是怎么回事。平方根倒数不是有个 magic number 算法吗？那个算法能比老老实实用浮点数算快很多（曾经如此，现在硬件上可能也是类似的 magic number 有优势），所以英伟达就把它写进去了，有专门一个函数算平方根倒数。

这个近似到底有多近似？研究浮点误差的话，对误差的衡量单位是 ULP。CPU 上大家熟知的 float 和 double，每个基本操作被要求在绝对精确的基础上做四舍五入，浮点误差是半个 ULP；而 GPU 的 SFU 算的是近似值——实测离最优的四舍五入结果会差 1 或 2 个 ULP（具体是 0、1 还是 2，每个函数不一样）。如果你觉得一两个 ULP 的误差无伤大雅，可以直接走 SFU，获得一个并不满足 IEEE 754 标准的浮点结果（很多情况下够用了），开启 `-use_fast_math` 编译选项就会走这条路；如果你对精度要求非常高，GPU 会对这个粗略结果做一定次数的牛顿迭代，使它最终满足 0.5 ULP 的精确要求。

还有一个细节：右侧大图最下面有深蓝色的 "tex"（texture）——处理纹理这样的信息似乎是 GPU 的本职工作，但它跟我们的主题没有太大关系。

不得不提的是：SM 的计算器数量真的多，它的四分之一个 SM 里面有 16384 个寄存器，可能跟 CPU 的 L1 缓存是一个数量级的东西。可能这也是为什么 GPU 连个加法都要算这么慢的原因：它表面上在找寄存器，实际上就像 CPU 在找 L1 缓存一样——有四个周期的延迟很正常。

### 64 位计算

CUDA core 用于 32 位及更低精度的计算，那么 64 位怎么办？

- 64 位 int 或者说 long long：好办——本来就算软件实现，也就没几条指令就做过去了；
- 用 32 位模拟 float 的 64 位版本（double）是真的很慢。在 Blackwell 上，应该是从 Blackwell 开始砍了 double 的 CUDA core（double 的 CUDA core 曾经有不少的）。

## 十二、Latency hiding 与 occupancy

CUDA core 的运算速度非常快，但相对慢的是 GPU 的显存，延迟比较大。所以 warp 经常遇到 memory-bound 的情况——SM 开始"养"：它明明一个周期只能做四个 warp 的计算，却可能养着 64 个 warp；如果有一个 warp 阻塞了、想访问的内存来不了，它马上被换下去，调度器会换一个可以执行计算的 warp 上来接着跑。这样计算单元和内存带宽都能尽量保持满负载，相当于做了一个 pipeline 一样的东西，把延迟掩盖掉，同时保持高吞吐量。

**Occupancy** 是什么？它是实际活动的 warp 数除以最大支持的 warp 数，是 0 到 1 之间的数。之前说了 SM 的效率是通过"养"出来的，需要很多 warp 才能把内存延迟隐藏起来，所以通常来说高 occupancy 是比较好的事情。

右边这个图截自 CUDA 官方教程：最大支持 warp 数在哪？就是 maxThreadsPerMultiProcessor，2048 除以 32。它下面有一行灰色小字：很多 CUDA 代码用了很多的寄存器。使用很多寄存器有什么不好？因为 SM 里的寄存器是有限的——只有 16384 乘 4 个，用多了还是会用光。所以为了让 occupancy 高，我们希望一个 CUDA 线程不要用太多寄存器。

但这件事跟我们之前用的 restrict 是矛盾的：给指针加上 restrict 修饰之后，编译器倾向于把本来需要一步的内存访问的数据存到寄存器里——存到寄存器里当然是好事，但会降低 occupancy。所以这里有一个需要你自己衡量的事情：你到底是想要高的 occupancy，还是想要多的寄存器。

基于 occupancy 可以提出一个相对科学的调块长的方法——直接调用 CUDA API：

- `cudaOccupancyMaxPotentialBlockSize`：假设已经写了一个 kernel 叫 my_kernel，写 kernel 时 gridDim 和 blockDim 都没定，这个函数可以帮我们定下来。它返回的 block size 是**最大的可能 block size**——在满足前面列的寄存器条件、shared memory 条件等、所有硬件支持的情况下，一个 block 能开多大；min grid size 是说至少要发射多少个这种最大的 block 可以把整个设备的 SM 用满。这适用于确实不知道块长开多少的情况；
- 如果块长已经定了，最后一行那个函数（`cudaOccupancyMaxActiveBlocksPerMultiprocessor`）在给定块长的情况下，可以告诉你一个 SM 可以同时调度多少个 block——相当于帮你算了一下 occupancy。

## 十三、总结：优化方向

到这里 CUDA 的核心内容基本讲完了。总结优化方向，就这四点：

1. **最大化并行执行**：在软件层面告诉 GPU 我有很多很多具有并行潜力的计算；
2. **优化内存访问**：比如 shared memory 和大块的合并访问；
3. **最小化 warp divergence**；
4. **利用 tensor core**（这次没讲）：它可以用来高效地做矩阵乘法。

## 十四、延伸阅读

下面这些比较常用的代码就不展开讲了，大家课后自己看：

- CUDA API 会返回错误，可以写错误处理；
- 要 benchmark 一个 kernel，可以在代码里写计时的小程序，当然也可以用 profiler；
- **Stream** 是让 GPU 并行执行 kernel 的工具。需要强调：kernel 一启动之后 CPU 是不会阻塞的，这使得发射第二个 kernel 成为可能——但仅仅是可能：如果你没有用 stream，kernel 2 在 GPU 侧是会阻塞的（虽然 CPU 语句可以接着跑下去，但 GPU 那一侧并不会真的跑起来，直到 kernel 1 运行结束）；如果你指定了两个 stream，这两个 kernel 在 GPU 端也不会互相阻塞；
- **统一内存（unified memory）**：让 CPU 与 GPU 的 memory copy 交给系统层面自动进行；
- **多卡编程**主要有两种思路：一种类似于 message passing——通过 cudaMemcpy 做信息传递；另一种类似于 shared memory——通过 enable peer access，让同一个指针可以在两台 GPU 上访问。

我的分享就到这里，感谢大家一个小时的辛苦付出。

---

## 附：听辨存疑与说明

1. 通义听悟逐字稿中有数段 ASR 乱码（warp 定义、64 位计算、occupancy API、stream 等片段），正文已根据上下文恢复语义，涉及的代码细节以官方文档为准。
2. 讲者口述的"从 CC 7.0 起每个 thread 有自己的 program counter"按原话保留（对应 Volta 引入独立线程调度）。
3. "Blackwell 开始砍 double 的 CUDA core"按原话保留。
4. 除法吞吐（约 1/8）、SFU 倒数（每周期 16 个、仅 32 位）、误差（1~2 ULP / 0.5 ULP）、128 个加法/周期、16384 寄存器等数值均按讲者口述保留，具体型号（Blackwell Ultra）以官方数据为准。
