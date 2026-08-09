# 1.2 Triton/TileLang Tile Level Programming —— 完整讲课转录

## 视频信息

- 标题：北京大学未名超算队 × LCPU AI Infra Seminars 系列讲座：1.2-Triton/TileLang Tile Level Programming
- 链接：https://www.bilibili.com/video/BV1Sb3w6nE8v
- UP 主：北京大学Linux俱乐部
- 时长：45:22
- 作业仓库：https://github.com/lcpu-club/wmhpc-training-camp-x-lcpu-ai-infra-seminars
- 活动官网：https://infra.seminars.lcpu.dev

## 校订说明

本转录以 B 站 ai-zh 自动字幕为底稿，交叉参考 ai-en 字幕，按语义校订整理。视频为中文讲授，正文沿用中文。ai-zh 字幕覆盖全片（末条时间戳 00:45:19.800，视频时长 45:22.6），未重新运行 ASR。原始带时间戳字幕保留在 `course-lecture-builder/jobs/BV1Sb3w6nE8v/raw/lectures/captions/`。

---

## 开场

第三部分由我来给大家介绍 Triton 和 TileLang——两种 tile level programming 相关的语言。这一部分的内容不是特别多，主要是希望大家建立一个 tile level programming 的具体思想，而不是很详细地讲解语言的细节。我们会大概对这两个 DSL 的思想进行表述，这部分尽量快一点讲，不耽误大家后面的时间。

## 一、复习：CUDA kernel 是如何映射到 GPU 上的

### 任务划分（执行的层次）

一个 kernel 启动之后，硬件按以下的层次把并行任务铺开：

1. **kernel** 启动，生成一个 **grid**；
2. grid 被划分成多个独立调度的 **block**；
3. 每个 block 包含多个 **thread**，thread 以 **warp**（每 32 个线程）为单位执行，同一个 warp 的 32 个线程同步执行同一条指令。

可以注意到：grid 是逻辑上的全部任务；block 是真正被调度到 SM 上的单位；thread 是最小的干活单位。

### 数据与内存的层次

内存层次分为三部分：

- **global memory**：最大的内存，容量大、延迟高，所有 block 都可以访问；
- **shared memory**：每个 block 内共享的内存，延迟较低，一般用于数据复用；
- **register**：每个线程单独自己的寄存器，访问最快，但一般只能存当前正在算的值。

这三个内存从上到下越往下越快、容量越小、越私有。

### CUDA 的问题

写 CUDA kernel 时，所有这些过程每一条都需要在代码里手动指定：

- 必须手动完成每个 block 和 thread 各自负责哪一部分数据（用 block/thread id 去算下标）；
- 必须手动指定数据在三个内存层次之间怎么搬运；
- 还要考虑同步问题、边界问题，一个一个手动判断。

总体上，从数据分块到线程执行的映射全部要自己写出来。CUDA 的优势是把一切可以控制性能的东西都交给你控制，但无形中也增加了工作量和代码复杂度。在性能要求没有那么极致的情况下，我们可以尝试一些更加省心的写法——因此引入 tile level programming。

## 二、Tile Level Programming 的思想

之前的 SIMT 编程是考虑每个线程做什么，线程写作和数据复用都要手动管理，写出来的代码比较繁琐、难以优化。回顾之前提到的几个 CUDA kernel，主要思想都是把数据分块、对每一块进行操作。我们能不能从这个角度去写代码？

tile level programming 就是换一个角度：不再考虑每个线程写什么，而是把每个 block 内线程的工作抽象成一个对整体分块（tile）的操作。原来在 CUDA 中"加载一部分数据 → 计算 → 写回"这些工作，在 tile level programming 中都省略了手动管理线程的部分，由编译器负责把这些 tile 的操作自动映射到 block 内的具体线程。我们不再需要管理线程级别的细节，只需要对整个分块做块级操作，简化了实现难度。

大致流程：写 kernel 时，整体都是在一个 tile 上操作——先知道当前 tile 的位置，从 global memory 把要算的这一块加载过来，计算之后再写回去。边界处理不一定是最后一步，大概率夹杂在计算过程中，需要根据不同代码自行处理，所以列在最后。

**DSL** 的意思是领域特定语言：基于一种通用语言（如 Python、C++）设计的为特定领域服务的语言，比如 SQL 之于数据库、正则表达式之于文本匹配；CUDA 以及今天要讲的 Triton 和 TileLang 也都是 DSL。

## 三、Triton

### Triton 与 CUDA 最根本的区别

CUDA kernel 从 thread 出发、自下而上构建：先决定每个 thread 做什么，再决定每个 block 做什么，最后是整体做什么。而 Triton 的基本单位是 **program**，可以理解为一个分块过程：每个 program 负责一个数据块。它是从块开始、自上而下的过程——我们只需要描述每个 tile 的位置和形状，再对 tile 整体操作，由编译器负责把 tile 内的计算映射到 thread 和 warp 上。

Triton 保留了数据和如何分块的决策权，把线程级映射的权利交给了编译器。一个 Triton program 近似对应一个 CUDA 的 block（CTA）；program id 对应 block id；thread id 不再显式出现。

### 向量加法的例子

用代码解释 Triton 具体怎么写的，先看一个比较简单的向量加法。

写 kernel 分两部分：一部分是 GPU 端运行的 kernel 代码，另一部分是 CPU（host）端的入口，由入口调用 GPU kernel。

CPU 入口：

- import Triton，把 `triton.language` 简写成 `tl`；
- 定义函数，把 X、Y 相加得到 Z，N 是向量长度；
- 定义块长（block size），这里手写为 1024——这个值可以灵活调整，根据机器和实际效果手动调优；
- grid 是分块后的块数量，即 N / block_size；
- 调用 kernel 时用方括号：`kernel[grid](...)`，方括号里是分块数量，后面是正常传入的参数（x, y, z, n, block_size）；grid 会把函数分成若干部分在不同地方并行调用。

GPU kernel：

- 和 CUDA 一样需要声明这是 GPU 上的 kernel 函数，Triton 里是在函数上写 `@triton.jit` 装饰器（`@` 这一行），编译器就知道要把这部分编译成 GPU 函数；
- `tl.constexpr` 与 C++ 中的 constexpr 类似，表示 N、block_size 是常量；
- 中间的流程比较固定，写大部分 kernel 都需要包含：
  1. 取出关键值：首先是 `pid = tl.program_id(0)`——当前 program 是第几个，0 是维度的意思；向量加法只有一维，所以只写 0；矩阵乘法分块是二维坐标，横坐标写 0、纵坐标写 1；
  2. 取出 offset：当前块在原始数组里的位置（编号 × block_size，加上 0 到 block_size 之间的下标集合）；
  3. 取出 mask：判断合法的状态量，把 offset 中小于 N 的部分拿出来，屏蔽掉 N 以外的部分（最后一块可能不是整块）；
- 计算：先把 X、Y load 过来（`tl.load(X + offset, mask=mask, other=0.0)`），mask 是屏蔽语法，other=0.0 表示对屏蔽掉的部分用 0.0 填充（相当于 padding），然后直接相加，再 `tl.store` 写回全局内存。

可以发现我们的操作完全建立在整块的基础上：把 X、Y 在当前块的整个向量取出来直接算，避免了繁琐的手写过程，更像是对向量整体操作而不是逐个元素操作——这就是 tile programming 的主要思想。

### 矩阵乘法的分块思路

向量加法简单，因为两个一维向量可以直接切段并行；矩阵乘法乍一看分块方法不是一眼能看出来的——如果你在 A 里选了一块，它对 C 的影响其实是比较模糊的，拆 A、B 不容易拆出互不相关的部分。

所以从 C 入手：结果矩阵 C 的每个位置和其他位置没有关系，可以拆分 C 的一个个部分。先分出一块 C（比如深蓝色的块），它由 A 的某几行和 B 的某几列贡献；把这两部分单独拿出来乘。但矩阵可能很大，行向量和列向量仍然太大，不是好的分块。于是对这两个矩阵再分块：先对 M、N 分块之后，第三次对中间循环变量 K 再分块。最终答案由 K 方向每一块 A 和对应 B 的乘积累加得到。

### 矩阵乘法的代码

与向量加法相同的部分直接跳过。矩阵常量 A、B、C 的 M、N、K，以及三次分块的块长 block_M、block_N、block_K。

**stride（步长）的作用**：矩阵是二维数组，被加载到内存时平展成一维向量，所以需要用步长方便地访问行和列。例如 stride_am 是 A 每增加一行需要的步长（即 K），stride_ak 是 A 每增加一列需要的步长（即 1）。这些步长不需要手动计算，Triton 提供了 `.stride()` 方法自动确定矩阵的步长，0 和 1 同样表示维度的区别（竖着数和横着数），把步长带进去即可。

**pid 与 offset**：分块之后块的标号不是一维的 123456，而是一个二维坐标（第几行的块、第几列的块），所以需要同时知道行编号和列编号；offset 用于计算当前块有效的（一维）位置，用两个一维拼出一个二维位置。

**累加器**：定义 ACC，大小是 block_M × block_N，类型 float32。在计算一个块的过程中，把 A、B 对 K 分块的所有答案累加到同一个位置。

**K 循环（第三次分块）**：CPU 函数里只分了 C 的块，A、B 的 K 分块是在 kernel 内部完成的。在 kernel 里手动写：求 A、B 的位置（如 A 的 offset_m × stride_am 把挨着的东西分散开、变成一列，K0 + offset_k 是单独 K 块的长度，K0 枚举开始分块的位置、每次加 block_k），取好下标后 load 进来（mask 判断不能超过 M 和 K），然后直接用 `tl.dot` 把两个块乘起来放进 ACC。细节（矩阵内部怎么算）都由编译器完成。

之后如法炮制：获得 C 的位置和合法性要求，把答案 `tl.store` 存回去。

整体流程和向量加法一模一样，只是矩阵乘法稍微复杂一些。每次写的时候需要拿出来的量是固定的、流程差不多，只有中间计算部分有差异——都是整体对一块进行计算，而不是把块里面再拆分。

### Triton 的局限性

（注：上面的 GEMM 写法只是一个教学例子，真要写高性能 kernel 还有很多细节要改、很多地方可以优化。）

1. **编译器是黑盒**：你无法手动管理内存访问和布局，这些全部交给编译器。CUDA 里可以一行行手动控制，Triton 里只能通过调 block_size 等参数影响结果，无法指定"把两块数据错开存放"这类细节；
2. **看不见生成的汇编**：kernel 慢的时候无法判断源码和生成的汇编之间的关系；profile 之后也只能看到编译后的东西，很难定位原始代码的问题出在哪；
3. **性能天花板受限**：编译器自动改，很多东西改不了，性能会和 CUDA 有差距；比较复杂的 pipeline 之类 Triton 无法表达，复杂 kernel 写起来也比较别扭。

总体来说，Triton 放弃了一部分人的控制权，换来较高的开发效率；当细节控制成为瓶颈、或者要追求非常高性能的 kernel 时，就需要其他工具——比如 TileLang，或者回归更底层的 CUDA。

## 四、TileLang 的编程模型

为什么引入 TileLang？为了躲开前面的问题：Triton 太简单、比较黑盒，看不见内存层次；CUDA 又太复杂。TileLang 的核心主张是在中间取一个平均：**更强调显式地把层次结构写出来**——把数据在哪一层、怎么流动这些影响性能的事情明确写进代码，但线程级别的事情仍然交给编译器处理。

所以 TileLang 代码更像是在描述一个数据流：global memory、shared memory、按维度分块的循环，都是一层一层写出来的，形成一条显式的数据流水线（从 global memory 取过来、算完、存回去）。

### 三层抽象

TileLang 核心设计模型是三层不同程度的抽象，每层对应不同的控制力度和适用场景：

1. **第一层（高抽象）**：内存和并行都是隐式控制，大部分东西自动调度——没有很显式地体现 TileLang 的优势区间；
2. **第二层（tile level）**：最常用的视角，显式写出并行和内存控制，获得比较好的性能；
3. **第三层（细粒度）**：不太常用，把 thread、warp 等和 CUDA 一样细致的控制也写进去——失去了简洁的特点。

对比：Triton 更接近一个向量表达式——写一个 tile 后，它落在 shared memory、寄存器还是中转一下，都不是你写的，由编译器决定；TileLang 更接近手写 CUDA kernel 的结构——你要明确写出"把它放进 shared memory、再从 shared memory 开始算"，只是不需要考虑线程和同步的问题。换句话说，Triton 让你直接描述"算什么"，TileLang 让你描述"怎么在内存层次里把东西串起来"，更可控；而高性能 kernel 的数据流动方式往往决定了性能本身。

## 五、CUDA 与 TileLang 的对照

左右两边做的是同一件事（同一个矩阵乘法），只是不同表达方式（不是严格的逐行翻译）：

1. **任务划分**：CUDA 用 block id 确定当前 block 负责哪一块输出、用 thread id 确定每个线程负责哪个元素，需要显式算出行列；TileLang 的 `T.kernel` 直接定义了 CTA 的 grid，BX、BY 表示当前 CTA 负责的输出 tile，thread id 不会一一对应，数据和 thread 的映射由编译器完成；
2. **存储**：CUDA 用 `__shared__` 显式分配 shared memory 缓存当前 K 分块的 A 和 B；TileLang 用 `T.alloc_shared`——虽然抽象层级提高了，但仍然要手动把数据放到 shared memory 中；ACC 累加器两边一样；
3. **沿 K 分块的迭代**：CUDA 用普通 for 循环，每次处理宽度为 BK 的一块；TileLang 用 `T.Pipelined` 表达式——它不仅是循环，还告诉编译器这是流水化的：算上一块时就提前加载下一块，避免机器空转、提高效率，包含额外的调度信息；
4. **数据搬运**：CUDA 需要手动对每个线程计算地址、判断越界、把元素一个一个写进 shared memory，最后还要同步一次保证整个 tile 加载完成；TileLang 把整个数组的线程级操作概括为 `T.copy`——描述从 global memory 哪个 tile 搬到哪个 shared memory tile 即可，合并缓存和同步都交给编译器；
5. **计算**：CUDA 用内层循环让每个线程不断读 shared memory、更新自己的 ACC；TileLang 直接用 `T.gemm`——一个更概括的语义，编译器可以进一步把它映射到 tensor core 或其他更合适的硬件指令；
6. **第二个同步**：CUDA 循环末尾需要再同步，保证当前块用完后才覆盖 shared memory；TileLang 中这类事情和 pipeline 一起由编译器组织，不需要单独对应；
7. **写回**：两边都是一样，把结果写回，TileLang 直接 `T.copy`。

概括：CUDA 把线程索引等线程级细节逐步展开；TileLang 只保留内存层次划分、数据搬运和计算结构，把大量重复的线程级操作交给 primitive 和编译器。它写起来比 CUDA 简洁，又没有 Triton 那么不可控。

## 六、TileLang 常见的 primitives

（以下只列对示例代码的解释，更详细的部分可以课后看 TileLang 文档。）

1. **定义与启动**：声明它是一个 kernel（`T.kernel` 装饰器），定义一个 TileLang 函数；创建 grid 之后，一个实例对应一个 CTA（线程写作数组，可以理解为 block）；
2. **内存相关**：`T.Tensor` 代表从 global memory 拿东西；`T.alloc_shared` 是 shared memory；`T.alloc_fragment` 是寄存器——每个线程自己寄存器中的计算结果；
3. **数据流**：数据从 global memory 来，先用 `T.copy` 把它放到 shared memory，再计算；计算的结果在寄存器（fragment）里，最后再用 `T.copy` 放回 global memory；
4. **T.Pipelined**：流水线，让缓存和计算重叠，可以隐藏 global memory 的高延迟；
5. **T.gemm、T.copy、T.reduce_sum** 等。

完整的 TileLang GEMM 代码流程：把参数写出来，声明 kernel；先把东西从 global memory 拿出来，开始循环（先算竖着分块的数量、横着分块的数量，声明线程数），从 shared memory 加载 A、B 的结果放到 C 的 local（fragment，因为结果放在单个线程的寄存器里），type 就是前面定义的 float16 简写；先 `T.clear` 清空累加器，然后按 pipeline 计算，最后把答案 copy 回去。

## 七、总结：三种写法怎么选

回到最开始的问题：GPU kernel 有 CUDA C++、Triton、TileLang 三种写法。它们没有包含关系，是侧重点不同的一条线上的三个点：

- **CUDA C++**：控制力度最细，适合极端的性能调优和特殊硬件机制；但对单独硬件的特化泛化性稍差，换一张卡性能可能受影响；
- **Triton**：简单，可以快速写出简洁的 kernel，比如逐元素 kernel（向量相加、逐元素相乘等）；但控制力弱；
- **TileLang**：两者的结合者——显式表达内存层次和数据复用，保留了 CUDA 的结构，比 Triton 控制力强一些，适合搞清楚一个高性能 kernel 的内存结构；可移植性更强。

选择取决于瓶颈是什么：想写得容易、算子不奇怪，直接用 Triton；要对特定硬件做特化，用 CUDA 更合适；如果瓶颈是性能、而且性能取决于内存层次的数据搬运和 shared memory（大部分 kernel 的性能卡在这里），用 TileLang 把数据流显式写出来更有利于优化。

以上就是我今天分享的部分，谢谢大家。
