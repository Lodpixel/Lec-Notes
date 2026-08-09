# W.1 Parallel Programming with TileLang —— 完整讲课转录

## 视频信息

- 标题：北京大学未名超算队 × LCPU AI Infra Seminars 系列讲座：W.1-Parallel Programming with TileLang
- 链接：https://www.bilibili.com/video/BV1rhMD6YEKM
- UP 主：北京大学Linux俱乐部
- 主讲人：孔昊然（Infra 方向博士生）
- 时长：1:26:31
- 活动背景：2026 年暑期 Weiming HPC Training Camp × LCPU AI Infra Seminars 第三次活动录播

## 校订说明

本转录以 B 站 ai-zh 自动字幕为底稿，交叉参考 ai-en 字幕，按语义校订整理。视频为中文讲授，正文沿用中文。ai-zh 字幕覆盖全片（末条时间戳 01:26:30.660，视频时长 86:30.7），未重新运行 ASR。原始带时间戳字幕保留在 `raw/lectures/captions/`；讲者口语中的专有名词（TileLang、TIRX、FFI、CUTLASS DSL、TMA、wgmma 等）按上下文恢复，个别听辨不清处已标注。按讲者要求，转录以讲解当时的 TileLang 版本为准，后续如有更新以源码为准。

---

## 开场：回顾与课程目标

大家晚上好。我们首先回忆一下第一次对 TileLang 的介绍（逸飞的 PPT），当时讲得比较简单，主要就是三页。如果我的 PPT 有刊误或者讲得不准确，大家可以开麦直接说，我们之后再讨论，最后把 PPT 整理成一版没有任何问题的参考资料。

另外要说明一点：DSL 的进步在过去几个月里非常快，所以我们照现在的版本讲解，之后如果发生了更新，肯定以源码为准。

### 回顾 Session 1 的三页 slides

第一页我们做了最简单的 GEMM 映射，主要介绍 TileLang 里的一些 primitives：

- `T.alloc_shared`：分配 shared memory；
- `T.alloc_fragment`：分配 fragment（这个抽象挺有趣，后面会详细展开）；
- `T.copy`、`T.gemm`、`T.Pipelined`：用这几个 primitive 就可以表达一个比较复杂的 MatMul。

高度简化的数据流是：从 global memory copy 到 shared memory → 在 shared memory 上完成一次 GEMM → 结果存到寄存器的 fragment 中 → 最后存回 global memory。

从架构的角度看：tensor core（或者说计算能力）每一代都在翻倍，但数据搬运能力想打满比较困难，所以有很多技巧来完成这个更复杂的事。用 TileLang 写代码，可以相对屏蔽掉复杂的硬件相关的东西，对表达算法很有好处。

MatMul 的源码我们就快速过一下，之后会对每个 API 做更深入的分析，但大家一定要明白数据流是什么样的：输入在 shared memory 上、输出在 fragment（寄存器）上，通过 pipeline、copy、gemm 完成 global → shared → 计算 → global 的过程。

## 一、对上一次关于 bank conflict 的纠正

上一次学弟讲到 bank conflict，当时有人提问什么情况下会出现 broadcast。那张图可能带来一些误会：**并不是"32 个线程访问一个 bank"才会出现 broadcast**。

从硬件角度理解：可以把每个 bank 看作一个独立的小型 SRAM，32 个 bank 可以并行工作，所以它们可以给这些 thread 提供（不同）地址。但每个 bank 一次只能响应一个访问请求：如果你访问的地址正好映射到相同的 bank 上（且不是同一地址），就会出现 bank conflict，访问变慢。

broadcast 的关键点在于**相同的地址**：bank 在同一周期内无法响应多个不同地址的请求，但如果多个 thread 访问的是相同地址，它就可以直接 broadcast 出去。

举例：16 个线程读 shared[0]、16 个线程读 shared[1]，会以 multicast 的形式完成这次 shared memory 访问，不会出现 bank conflict。甚至可以写一个例子：每两个线程读相同的地址，就会变成 16 个 broadcast，一次 multicast 出去，也不会有 bank conflict。大家自己写个小的 warp 例子很容易验证，所以这里对上次的介绍做一个刊误。

## 二、CUTLASS DSL 与 TileLang

我们在推送里提到 NVIDIA 推的一个 DSL，叫 CUTLASS DSL（slides 来自去年 11 月 NV Open Day 的介绍）。它和 TileLang 某种程度上是可以一起使用的，但 TileLang 使用起来会灵活得多。

这张 PPT 信息量有点大，一下子放出来大家可能会懵，但后面会慢慢介绍。里面有一些很关键的抽象，比如独立的 IR。现在 TileLang 的 IR 抽象是 TIRX（如果关心陈天琪的话，会发现他们今年 6 月推了一个教程讲 TIRX；去年还是 TIR）。

从编译原理的角度：我们把优化叫做 pass，不同的 IR 之间做转换，在转换中引入优化。这个思想是通用的：把程序转成 IR，在 IR 之间一遍一遍加入我们想要的变化，从而在性能和抽象之间取得平衡。

这里强调：我们要显式地控制 memory allocation 和 movement。在最简单的例子里，我们要在最前面分配 shared memory 和 fragment。TileLang 还提供细粒度的注入：如果你是专家，可以用很犀利的 primitive 直接操作程序（第三层）；但过多引入只有专家才需要的手动操作，DSL 也就失去了优势——我们是想用比较少的 effort 拿到比较好的性能。

所以今天的重点在第二层：以 developer 的视角，把 TileLang 主要当做一个 tile library，它提供很好的 primitives 帮我们写 tile program、拿到不错的性能，尤其是在 memory-bound 的情况下减少工作量。

TileLang 还有一个非常大的优势：底下可以贴到不同的后端（这里提到 NVIDIA 和 AMD），支持各种各样的硬件；理论上如果有大哥愿意做各种国产卡的支持，也都是可以实现的。所以 TileLang 是一个相对非常灵活的语言。

### copy 和 gemm 能生成多复杂的 IR

CUTLASS DSL 的五个例子我们不会逐行讲，但映射关系是清晰的：我们写的 `alloc_shared`、`alloc_fragment` 会变成怎样的 TIR。copy 和 gemm 虽然在高层的写法很简单，但可以生成相对复杂的 IR：根据 SM 的计算能力（SM80、SM90 还是 SM100），只要完成了 lowering 支持，程序员只需要写 copy 和 gemm，它就可以生成比较先进的指令——比如 SM80 生成 cp.async，SM90 生成 TMA。

TMA 的详细介绍、以及它里面有些避免 bank conflict 的办法，会在明天（下一讲）介绍。这里演示的是写高层 TileLang 代码，但它在之后会站出各种各样的 TIR，是一个相对复杂的编译流程，我们可以在流程中灵活地加入需要的东西。

去年 11 月生成的 CUTLASS DSL example，这七个月里 TileLang 也发生了非常大的变化，但类似 GEMM 那种 `program` + `main loop` + `epilogue` 的结构可以很好地生成出来；SM90（Hopper）上的 FP8 GEMM 也可以比较好地实现。具体的 tensor core 介绍留到明天。

## 三、课程主线：TileLang 帮我们减少了什么负担

通过上面两个例子（Session 1 的 GEMM 介绍和 CUTLASS DSL 讲座）可以看出，TileLang 帮我们隐藏了一些 effort。但在写算子这个任务里，肯定还有很多东西需要我们自己决定、编译器无法完成——不仅有很上层的算法变化，还有一些 IR 难以隐藏的东西。这部分就是我们今天要详细关注的。

TileLang 是一个让我们减少心智负担的工具，但我们要很清晰地理解：它到底能帮我们减少什么负担，以及还有什么东西需要我们仍然关注。

主线按 TileLang 的 feature 来讲（为了讲清楚，我准备了几个 kernel，如果没时间演示，会把代码发给大家自己跑）：

1. JIT 与 lowering 的过程，以及 TileLang FFI；
2. fragment、parallel 与 layout inference；
3. tile library primitives；
4. pipeline API，以及从架构角度理解它为什么能带来更好的性能；
5. pipeline 与 persistent 的对比（两者某种程度上可以互相转化，什么时候选哪个）；
6. mega kernel（过去一段时间非常火热的 topic，我会以我的视角给出一些 insight）；
7. profiling：补充上次 session 的 NCU。CUTLASS DSL 里率先引入了一个叫 in-kernel profiler（IKET）的 kernel 内 profiling 方式，TileLang 里也 experimental 支持了；另外还有 Nsight Systems（NCS）做 system 层面的分析。在大模型时代，我们也可以直接让大模型读 SASS 来分析算子。

## 四、写算子在 architecture 层面要处理的四个问题

我们认为，写算子在 architecture 层面需要处理四个问题，以及 TileLang 到底帮我们隐藏了什么：

1. **每个 CTA/thread 到底负责处理哪些元素；**
2. **操作的数据位于 global、shared 还是 register**（这里不提更多的 memory hierarchy 复杂度，比如我们之后会用到 B300 上的 TMA，先不引入）；
3. **load、compute、store 如何排序和重叠**（这里的 load/store 是从内存语义角度说的；如果考虑分布式场景，remote 数据可能要通过通信拿到，原理相通——我们希望把计算和数据搬运重叠起来）；
4. **什么样的参数最好**：thread 数、stages、CTA 数量怎么开（这需要一个 tuning 的过程，很难在一个很大的搜索空间里直接给出最优答案，搜索问题一贯是学术界喜欢讨论的东西，我了解得也不深入）。

今天我们主要讲前三个问题。TileLang 有一个非常好的性质：我们一定程度上只需要考虑 CTA 到底负责什么 workload，它有一些自动推导的 inference 机制去决定 thread 到底负责哪些工作——这对编程和心智负担都很友好。

另外 memory scope 需要我们来选，但编译器也可以帮我们推导这些映射和搬运。这里与 CUTLASS DSL 不同：CUTLASS 的 cute layout 是一个非常严谨的代数体系，可以显式计算这些映射；而 TileLang 非常灵活——如果觉得映射需要 annotate，也可以在用户这一层做调整。

第三个就是看 `T.Pipelined` 是怎么做自动展开的。DSL 没做好的地方我们可能需要手动完成，比如复杂的依赖需要做一些 synchronize 来避免 race。

## 五、基本原语：prim_func、kernel、parallel 与 JIT

之前主要提到 GEMM，这里用一个 ADD 的 TileLang kernel 介绍三个原语：

- `T.prim_func`：函数声明；
- `T.kernel`：在这里声明 grid（CTA 数）和线程数；
- `T.parallel`：给出并行域；layout inference 会结合输入输出、循环和 tile operator 的约束，帮我们决定具体的 thread 执行什么。

其中一个很重要的 feature 是 JIT compiling（过去半年非常火热）。我给出的模式主要是 lazy JIT，但现在最新的版本好像是 eager JIT，主要是写法的不同，性能上应该差不多。JIT 的好处：我们可以把静态参数在首次调用的时候就 lowering 成一个 artifact，如果出现相同的静态配置就直接调用；此外我们可以对这个 artifact 导出很多东西，去查看我们想看的过程——它对我们来说不是一个黑盒。

JIT 本身是"提前全部编译好（AOT）"和运行时之间的权衡：静态参数让我们可以静态地做很多决定、完成很多推导，从而降低运行时 overhead；另一方面我们希望控制 artifact 的大小、提高复用率，避免运行时出现很大的编译开销。如果在运行时真的触发了编译，这个过程相对比较复杂，这里就不展开了，但思想上很清晰。

### 编译链条概览

用户看到的是 Python → TileLang DSL → TIRX（Tianqi 的 TBM/TIRX 也是很好的学习资料）。如果出现自动无法良好处理的情况，我们可以写 annotations；backend 可以接到 CUDA 和 CUTLASS DSL。架构层面有三类：SM80、SM90、SM100（当然可以有更多），中间有非常复杂的 IR 变换，其中有 layout inference（后面详细展开）、TIRX 还有很多 target-specific 的 pass——上层主要是语义表达，越往下越贴近硬件。在 backend 那边还分 host 和 device 两个部分，根据选择的 backend 有自己的一套编译流程。

最后着重介绍 TileLang FFI（TVM FFI 体系）。DeepSeek V4 的 tech report 里着重提到它可以把运行时开销、尤其是 kernel launch 开销降得很低。当 kernel 执行时间很短时，launch 就会变成 bottleneck，通过 TileLang FFI 可以在很大程度上降低它。当然 FFI 不是唯一的 backend，还有一些其他 backend，今天不讲。

### lowering trace

这个编译链条太长了，粗糙地提一下，大家能记住几个之后要展开的关键词就很好了。这里有一个非常好的文档/工具：lowering trace。可以直接拿到 output——做了什么操作之后变成什么样，都会有对应的 TIR 文件，这样我们可以判断它做的操作是否符合预期，然后去 play with it。推荐大家下去之后玩一玩，熟悉整套编译链条。

### TileLang FFI 的细节

参考去年的博客，简单来说 FFI 的 feature：

1. 可以把编译生成的 host 和 device 代码封装成一个可调用的 Python 对象，像调 Python 函数一样直接调 kernel；
2. 可以直接接受 PyTorch-compatible 的 tensor，很多元数据可以直接拿到，而不用在运行时手动处理，开销大概是微秒级——与一些很短的 kernel 执行时间相当，所以如果能节省这个时间，整体性能会非常 OK；
3. 可以把一些 Python 运行时的调用提前 lower 到 C++ 过程中，性能比较好。

DeepSeek V4 的 report 里优势列了很多，核心中的核心就是极低的调用开销，当然还有良好的兼容性。它明确写了 reducing 调用开销（CPU side overhead 被大大降低），主要用到的就是找 Tianqi 的这个 FFI framework。

### 再介绍 TIRX

去年 11 月的 PPT 里我提到 TIR，现在 TileLang 维护的 TIR 已经是 TIRX（TBM/TIRX）了。这也是一个很有趣的东西，Tianqi 写了一个很好的入门教程（MRCS 入门？），大家之后写必卡算子也可以参考。简单说：TileLang 的前端最终构造的是 TIRX 的 `prim_func`，在 IR module 里做 progressive lowering，做很多编译上的操作，最后生成 artifact。它肯定要依赖很多东西，包括对要执行代码的分析和硬件本身的限制。

## 六、T.kernel：定义 CTA workload

回到核心 primitive `T.kernel`。我们需要在这里决定：

- 一共有多少逻辑的 CTA；
- 每个 CTA 负责哪个 tile；
- 一共有多少个线程。

但线程具体做哪些事情，不用我们来分配——TileLang 不会替我们做分块，所以：

- 如果 CTA 的 tile 选小了，ILP 就不足（ILP 是提升 in-flight 的好东西），访存的开销就无法掩盖；
- 如果选太大，register/shared memory 的压力太高，occupancy 一看就不好；但 occupancy 并不代表性能（后面会详细介绍）。

我们拿到的抽象是：编译器自动做了 thread 内的映射，但 **workload decomposition 需要我们自己完成**。

## 七、fragment：CTA 层面的逻辑寄存器块

fragment 是 TileLang 提出的一个很好的抽象。thread-level 编程需要管理线程和每个线程拥有的寄存器；现在我们让 CTA 自己先把活分配给线程，就相应地需要把所有线程拥有的寄存器以编译的形式给到 CTA——这就是 fragment。

需要额外注意：fragment 不是每个线程私有的，而是分给一个 CTA、由 CTA 内所有线程持有的一个逻辑寄存器块。物理上的 scope 主要是 global memory、shared memory 和寄存器（寄存器对程序员一开始不可见），TileLang 里新增了一个 scope 叫 fragment。

## 八、Memory hierarchy 回顾（以 Hopper/H100 为例）

物理上有哪些 scope？以 Hopper 为例：

- 最外层是 HBM（global memory）；
- L2 对程序员是透明的，但读 DeepSeek report 的话，我们可以在 PTX 层面做一些缓存上的 hint（比如想办法提高 L2 locality）——我们不能手动操作它，但可以想办法优化这部分性能。

这是整个 GPU 的视角；写 kernel 的视角主要 focus 在 SM 上。H100 有 132 个 SM（物理限制）；NVLink 是之后通信课会提到的环节。先把视角 focus 在单卡、再 focus 在一个 SM 上：

- 一个 SM 分为 4 个 sub-partition，有 4 个 warp scheduler；
- Hopper 有一个特殊抽象 wgmma（warpgroup MMA），可以让 4 个 sub-partition 一起执行某些东西（今天不展开）；
- 每个 SM 的物理资源有限：不同精度的计算单元（CUDA core 等）、混合精度 tensor core（期待明天的 talk）、load/store 单元（LSU）和 SFU（special function unit）。

SFU 还闹出过有趣的新闻：B200 上 SFU 不够导致一些算子性能特别差，所以 B300 上专门做了 SFU 性能的增强（不是特别差，只是需要加强），因为 SFU 变成了 bottleneck；我们也可以在软件上做操作去掩盖 SFU 性能的不足。所以写 kernel 本质上是和硬件设计做 co-design 的过程。

存储上的资源：每个 sub-partition 有 register file，底下是 256KB 的 L1/shared memory cache，以及明天会提到的 TMA——它可以加快 memory 的搬运。这些资源由物理硬件决定，资源不足时就会遇到性能问题，所以每个高层 kernel 在硬件上执行时都要把这些资源考虑进去。

Software 视角：hierarchy 上主要是 grid、thread block、thread 这些更复杂的抽象，非常硬件相关。现在 thread-per-thread 的抽象在一定程度上被交给 TileLang 处理，我们需要在 CTA 层面把 register 和 local memory 的抽象拿过来，也就是 fragment。cluster 是 H100 上专有的 memory hierarchy 层次，再往上是大家共有的 grid 抽象。

这张图把 H100 的 memory hierarchy 画得很清楚（SM 1..N 是 132 个，两张图连在一起是一个又瘦又长的图，这里按中间展开成左右两部分）：4 个 warp scheduler、CUDA core/tensor core 计算单元、load-store 单元、register file、TMA 加速搬运、shared memory 与 L1 共享 256KB 硬件资源、global memory（80GB/40GB 级别）。后面提到的 kernel 主要也是在 H100 上写的。把这张图看明白，前面的 hierarchy 就全都清楚了。

## 九、T.parallel 与 layout inference

了解了 memory hierarchy 抽象之后，再回顾 Session 1 提到的 primitive `T.parallel`。我们描述逻辑的迭代域之后，layout inference 就会结合输入输出、循环和 tile operator 的约束，帮我们决定具体哪个 thread 执行什么。原本在 thread level 要考虑的很多复杂事情，被 layout inference 推理出来、自然解决。

layout inference 是编译期的约束求解，是 TileLang 非常精髓的东西，大家感兴趣可以详细了解。主要就是三种 layout：global memory 的 layout、shared memory 的 layout、fragment 的 layout（与 global 的无关，不能搞混）。在推导过程中，这个 feature 帮我们把 logical 的东西映射到 physical 的东西。

还会提到怎么在 shared memory 上通过 **padding** 的方式避免 bank conflict。之前已经重新解释过 bank conflict：它主要来自物理上的硬件限制——每个 SRAM 同一时间只能服务一个 address，所以我们要想办法让一次并行的 32 个 SRAM 都取不同 bank 上的地址；如果取相同地址就可以 multicast 出去。加 padding 会引入一些额外部分专门用来避免 bank conflict，其实还有一个更好的策略——swizzle，这部分也期待明天（下一讲）。

### Case study：transpose 算子

transpose 是一个非常经典的 memory-bound 算子。这里的 padding 做法是加一个 block K：如果不 padding，shared memory 访问本身就全是 32 的倍数，物理上就全是 bank conflict；加一小节 padding 后，物理访存拿到的是这个位置，逻辑上按正常下标访存访问的是这一列，对应到它右边的 bank——这样在访问一列的时候就避免了 bank conflict。

另外重温上上讲的 coalescing（合并访存）：访问 global memory 时，如果一次性访问一块连续的 global memory，可以一次提上来——4 个 sector 是 128 字节。所以对 global memory 的访问需要物理上连续；为了避免访问一列在 HBM 上性能很差，我们可以做分块。

## 十、需要用户手动处理的四件事

这里一共提四个 feature：CTA 的 workload、shared memory、loop 的 layout、同步。loop 的 layout 直接 refer code 就够了，不细说。

同步这块要提出来的原因是：如果 shared memory 上存在读写冲突（race），这个 race 是要我们手动处理的——不加 synchronize 编译期不会报错，但 race 会让代码变错。所以同步的边界还是需要手动表达：TileLang 可以帮我们做很多事情，但同步过程还是需要我们自己的。

## 十一、Tile library primitives

回到 3.1：tile library primitives。和 layout inference 一样，这两个 feature 是 TileLang 在 tile-level programming 特别好用的原因：它提供了很多 tile-level 好用的 API/封装函数。

比如 `reduce_sum`：直接调一个 reduce_sum 就可以省掉很多"怎么 reduce、怎么在 thread level 做 reduce"的心智负担。它有点像 C++ 里的模板/STL——实现很多常见功能时少了很多心智负担。

它的价值在于：数据移动、规约、矩阵计算都保留 tile region 语义，并与逻辑并行循环共同参与 layout inference。因为保留了高层信息，inference 的时候不会拿不到我们关心的信息。这里列举的就是 copy、reduce_sum、parallel。

**copy 为什么复杂？** 数据搬运有很多小 trick——到底用什么样的底层原语搬数据？TileLang 可以根据 scope、shape、alignment、target 和 schedule 把 `T.copy` lowering 成不同的搬运原语。右边那张图来自 GTC25 的 "maximum memory bandwidth" 优化 talk：并非所有 copy 都用 TMA 搬最好，这里有 throughput 与 latency 之间的 trade-off——大块、对齐的数据适合用 TMA 搬，但有很多数据没必要用或者没法用 TMA；不同硬件上也有不同的高性能搬运实现。所以我们直接用 `T.copy`，让 TileLang 帮我们解决选择原语的心智负担：我们只需要描述数据搬运，target-specific lowering 会选相对合适的原语来实现。如果觉得它没选对，那可能就是我们对这个 DSL 能做出贡献的地方。

**annotation。** 有时我们担心它自然的做法没做好，就需要做一些 annotation，而且这个 annotate 我印象中是强制的——它是一个声明，只要这么写了就按声明的来。例子是 E5M6 的量化：要把 8 个 12-bit 的数据打包进 3 个 uint32 中；为了让一个 thread 不需要和其他线程交换数据，我们需要这 8 个 E5M6 数据由一个 thread 持有，所以可以写 `annotate_layout`，避免需要打包的 8 个值分散到不同的线程里。

**reduce 的学问。** `reduce_sum` 这个 API 也有不少学问：不同场景下要选择最合适的 reduce。如果是比较少的 reduce（比如 8 个值求和），让一个 thread 来做就够了；reduce 的场景差异会影响你选什么实现。

另外专门提到 replication：有时我们想避免 atomic——局部求和完之后再最后加起来、贡献到大的 global sum 数组里，这是一个很常见的实现。其中一种方式就是用 `finalize_reducer` 这种形式来表达：它是一个语法糖，省去了你考虑内存层次之间操作的麻烦。

对比一下会更清晰：如果你有一段 local 的数组，要做一些计算，那么所有人各有一份自己私有的小数组会方便很多，这时就要调 replication 这个原语。在 reduce 层面用这个语法糖就少了很多麻烦。原理很清晰：把 partial sum 先自己算，算完贡献到一个 global 的地方；replication 就是避免集中 race 的一种方式。replication 不一定要绑定 reduce——如果你有其他相同性质的操作需要把 global 的东西拿过来，也有对应的接口。

`finalize_reducer` 具体做的事情：把 local 的 sum 加到 global 里。它具体做了什么语法糖？我感觉主要是同步——因为前面可能是异步地做求和（在 warp 内、几个 warp 之间、再到 CTA 之间），最后的顺序不一样，所以要同步一下；另外还有 layout 相关的处理。往一个 global memory 里做加法这件事本身就有 race，所以直接拿一个 API 避免了最后一步的 race。最后一步有没有计算？我觉得肯定有计算——相当于做了 local 的 reduce，再把他们最终 reduce 到 global 数组里，语义上是完成了这个操作；但实现的细节我没有仔细看 lowering 过程。

讨论中补充：`reduce_sum` 应该是 shared memory 的 reducer，处理的是 register 与 shared memory 之间的 race——一个是把持有的几个 register 加起来，另一个是把一个 warp 里的东西加起来、再把几个 warp（一个 CTA）里的东西加起来。另外 `alloc_block_reducer`/`alloc_final_reducer` 这一套会自动处理从 register 到 shared memory 的一些东西：在没有这套语法糖之前，我们要手动处理 shared memory 上加法的 race；现在它直接包起来，我们就不用管了。那个临时的 shared memory buffer 开多大也是编译器自动决定的（从语法糖角度看，它分配了一个 block size 大小的 shared memory；具体 lowering 成什么可以之后做个实验）。示例语义是每一行做 X 平方的 sum，每个 thread 贡献自己的 partial sum。有 race 的场景不局限于 reduce，所以有 replication 处理更一般的情况。

## 十二、Warp scheduler 与 occupancy

回忆一下 warp scheduler：H100 上有一个 warp scheduler 硬件，resident 的执行上下文全部驻留在片上，所以物理上限是每个 sub-partition 驻留 16 个 warp（4×16=64）。具体驻留几个要看 warp 需要的硬件资源。

因为这个性质，上下文切换非常轻量，在 software perspective 上是免费的：它直接 select 准备好的 warp——当一个 warp 出现 load 或者 store 时，硬件层面可以直接切到一个准备好的 warp 继续执行，访存 latency 就被隐藏了。CPU 的进程切换需要保存上下文、代价比较大；而 GPU 的 warp 上下文都 resident 住，切换很轻量。

这里有一个很重要的数值叫 occupancy：我们可以几乎零开销地在多个 warp 之间交替发射指令。程序员无法控制每个周期到底发射哪一个，但如果提高 occupancy，就等于给了 scheduler 更多的调度空间——当一个 warp 没准备好的时候，硬件会自动切到另一个准备好的 warp 执行。但这只是调度空间，不代表实际利用率。后面提到的 persistent 就是和它相对的：persistent 在 occupancy 上可能不是那么友好，但可以用其他操作掩盖访存开销。

个人理解，warp scheduler 的 latency hiding 本质上就是"时分复用"：多个 warp 驻留在硬件上，一个 warp 访存时另一个 warp 执行，完全由硬件调度。

occupancy 能驻留多少 warp 受很多硬件因素限制：kernel 本身用的 shared memory、register 量（硬件决定总量、kernel 决定用量）、block size 等。kernel 最后一定触达某个上限，导致并发度不能再增加。

## 十三、T.Pipelined：用户可见的流水

与 occupancy/warp scheduler 相对的，我觉得是 `T.Pipelined`：我们直接描述一个 iteration 的 producer/consumer 关系之后，它就可以在 TileLang 层次上生成一个 for loop + 中间 rolling buffer 状态 + 应用逻辑（app logic）的三明治结构。个人觉得 TileLang pipeline 相当于做了一个用户可见的流水控制流构造。

这段代码我们一开始就见到了，但没有关注一个超参：**num_stages**。默认是 3，但 3 当然不见得是最好的数值。num_stages 的作用是发起更多的数据提前 load：让 pipeline stages 更长，有更多数据 on the fly；代价是更多的 shared memory 和 register 被占住，occupancy 受限、block 的 ILP 也会变长。

到底多少个 num_stages 合适？可以简单搜索一下。我写了一个小 demo 去搜：结构就是上面提到的非常简单的结构；一开始如果做机械排（compilation）会有一个冷启动过程，compile 完之后很快跑出来，就能看到不同 num_stages 下的 latency。结论（在 H100 上）：当 num_stages 小于 4 的时候，latency 明显在下降；大于 4 的时候慢慢变多、甚至时间更长。所以 num_stages 是一个 trade-off：太小，on-the-fly 的数据搬运不够，无法掩盖访存开销；太大，占据更多寄存器与 shared memory，occupancy 下降、性能也不好。最佳值可以搜索，而且换一个硬件，最优 stages 一定不一样。

## 十四、Persistent kernel：与 pipeline 的对比

与 pipeline 相对应的是 persistent。这两者被分成两个概念讲区别可能意义不是特别大，但我还是稍微提一下：它们其实是不同维度的并行。

persistent 的做法：launch 的 CTA 数量小于等于 SM 数（一般等于）。它做的是 task 并行：launch 了一个 CTA 之后，有一个 worker 池，worker 去拿很多很多 task 来做——并不是每个 CTA 对应一个 task。它不是 launch 特别多的 CTA、用硬件自动切换来隐藏访存开销，而是手动设计整个流程怎么变快：worker 怎么处理 task 0 到 task P 到 task 2P（grid-stride 式），让这个过程没有性能问题。如果选择了这种方式，我们要处理的复杂度就在这个过程里。

如果 task 是非常规则的 grid-stride 行为，可能还没必要用 persistent。一个可能比较好的例子是 mega kernel：用 persistent kernel 可以更直接地控制很多东西：

- task 的领取和执行顺序（不受调度影响）；
- CTA 跨 task 的生命周期；
- shared/register state 是否跨 task 复用；
- producer-consumer face 的组织（可以少一些同步）；
- 不规则 task 的负载均衡策略；
- worker 数量与主流资源之间的关系。

因为如果让硬件 warp scheduler 帮我们做调度，我们没有任何控制能力；很多表现不够好的话，调整能力有限。当然 persistent 也有代价：工作队列尾部任务、长期 live state 的生命周期管理都是比较复杂的问题。总之不管选择什么方式，总会有复杂度需要处理。选择 persistent 可以有更强的 schedule control，但它并不是免费的性能——如果你的任务很规则，它可能拿不到性能收益，甚至会被自动变成别的（具体行为一切以 TileLang 本身的行为为准）。

### Case study：InstanceNorm backward

这里选的 kernel 直接来自 TileLang kernels（"TIKERNELS"）里的 InstanceNorm backward 实现。它的实现没有直接用 `T.persistent` 这个 API，而是直接 launch 了 9 个 persistent blocks。虽然没直接调用 API，但它的想法也是满足"让有限的 CTA 循环处理多个逻辑任务"——这样的写法对写代码的人心智负担更小、控制力更强。

对比结论：如果你是一个特别规则的 iteration 形式 workload，可能 pipeline 就够了；如果你需要相对复杂的跨 task 保存状态、自定义次序、或者有需要 serialize 的东西，可以考虑 persistent。mega kernel 在此不多讲——学界的很多 mega kernel 工作考虑的点和大家没有达成共识；总之一切以更快为准，能优化得更好就是 make sense 的。选什么 primitive，要从 lifetime、dependency 和任务本身的 workload 特征出发，而不是一定要选哪个更高端的 feature。

## 十五、Profiling：NCU、NCS 与 in-kernel profiler

更高级 feature 的背后就是 profiling。昨天（上一讲）介绍了用 NCU 做 profiling，可以看 occupancy、memory throughput、stall 等指标，告诉你 kernel 哪边可能存在问题。这些肯定很有用，我们不能说 in-kernel profiler 就替代了 NCU，但 NCU 的局限性越来越明显——kernel 复杂了之后，指标可能给不了有用的信息。

除了 kernel 层面，system 层面还有一个 NCS（Nsight Systems）profiler，可以分析 host-side launch overhead（今天我们甚至专门提到了 TileLang FFI），还可以看 kernel 的 timeline 和依赖关系、overlap 怎么样。

**大 kernel 能用 TileLang FFI 减少 host 开销吗？** CUTLASS DSL 是可以的——TileLang FFI 甚至（在去年 11 月的 roadmap 里）CUTLASS DSL 也提到要支持，所以这是一个良好的 feature。DSL 本身不是特别固定的静态东西，未来各 DSL 支持 FFI 降低 host-side launch overhead 是 make sense 的；如果有一些祖传的 CUDA，能不能用这个救一下？其实都支持——这个东西本质上就是把"空转"（launch 空洞）利用起来。

但我个人实测的感觉是：TileLang FFI 并没有比 torch 的那个快特别多。它主要优势可能是在编译的时候：比如你要对 kernel 做 JIT，如果用 torch 去做这个 binding，torch 会有一大堆 overhead，编译和链接会变慢——这是选 FFI 的一个比较大的原因。所以我个人觉得：大部分 launch-bound 的 kernel 都应该去重写、去 fusion，而不是指望某种 hack 对整个 kernel 的调优有比较大的影响。如果你有 launch overhead 的问题，各种路径都可以尝试；kernel fusion 听上去非常 make sense。如果 fusion 之后 register 利用或 shared memory 空间不够、occupancy 下降等，那就进入了 kernel 优化的场景，作为一个技术手段可以去尝试，但具体能达到什么效果要在自己的 case 上试了才知道。

**in-kernel profiler**：当时一个非常震惊全场的 feature——可以拿到 kernel 内部很多细粒度的信息。不过现在 TileLang 的 release 版本（0.1.12）里还没有支持。可以用它分析 kernel 的 bottleneck 在哪，拿到比 NCU 更细的 bottleneck 分析，协助优化。

**大模型读 SASS**：大模型现在读 SASS 没有那么困难，所以如果我们不确定 lowering 到底实现了什么效果，可以直接把 SASS dump 出来分析。比如今天我们对某个 reduce 的行为不太确定，可以直接看 lowering 过程。TileLang 可以直落到 CUDA C++（backend 是我们可以自己选的，CUDA C++ 是一个可行的 backend），这样你都不用看 SASS。不过看 SASS 想直接让模型分析问题并改 PTX 去修复，还真不见得那么容易——PTX 变成 SASS 这个过程不一定对，也需要探索。但如果你思考的是"加入了一个新的 IR 优化之后，到底有没有对结果产生影响"，TileLang 可能更优秀一些：可以在任意地方看它做得怎么样。对性能有影响的质量你全都能拿到手，就没有什么黑盒。

补充：lowering trace（TileLang IR lowering trace）现在做得非常好：跑完之后会有一个示意图展示每个 pass 前后发生了什么变化、最后所有 pass 出完之后的 CUDA C++ 长什么样；而且支持修改之后 recompile——你可以拿到 C++ 或者在中间任意一个地方 edit，edit 完再重新跑、重新 compile，看性能怎么样。这是特别推荐大家回去尝试的事情。我们说的编译链条只是一种划分方式，它具体做的事非常复杂，通过 lowering trace 可以直接看每个 pass 在做什么，不用接收我这些可能有噪音的信息。

## 十六、参考资料与结论

参考资料：

- 一个之前开源的项目 tile-notes：用 TileLang 写了一些算子包（forward/backward、transpose 等），里面用的是 lazy JIT 写法，语法不一定是最新的，但写法可以学习；
- TileLang 官方 example；
- 我的一些结论基于 Hopper，换了不同硬件，编译过程非常硬件相关（比如 stages 最优值、版本变化），分享主要基于 TileLang 的特定版本来讲。

结论：**memory-bound 的 kernel 用 TileLang 写出来表现很好，有效带宽可以达到 75%～90%**，即使你写 CUDA C 也不一定会更好；如果有高手想使劲优化，也可以比一比。不过如果想提一个非常 general 的优化 IR pass，事情就变得复杂很多。

很多图片来自 NVIDIA 的 GTC talk 和 Open Day 分享；如果它们过时了——硬件变化相对较慢，但 DSL 过去几个月变化非常快——大家有怀疑的地方，可能是我没想对，也可能是行为已经变了，要以最新的 DSL 行为为准。

## 十七、总结（最后 3 分钟回顾）

现在大家再看这张图就非常清晰了：声明一个 `T.kernel`，用详细介绍过的 API 完成一个高层的 DSL 描述，它自动 lowering 成不同的 CUDA code——明显的 CUDA code 比较简单，也可以 lowering 成更复杂的 code。

对 bank conflict 的理解需要明确：它是基于 address 的（multicast/broadcast 行为），如果还是没太明白，直接写点 demo 测一下——bank 的行为就是硬件限制。

今天的分享主要基于 tile program 这一层的抽象，看了 TileLang 帮我们承担了哪一部分心智开销，还有什么部分仍然需要做决策。关键 feature 再重温：

- JIT（lazy JIT 写法，eager JIT 可能是更更新的写法）；
- 编译链条很长，但有一些特别好的 feature：TIRX、backend 可贴 CUDA 或 CUTLASS DSL、layout inference 自动推导 CTA 到 thread 的映射和 memory layout 变化、一系列 IR 优化实现不错的性能；
- CPU 侧有 TileLang FFI 优化降低 launch overhead——但能不能在 launch-bound 的情况下拿到满意的收益，待会儿需要再优化和尝试；
- 特别推荐 lowering trace，去看每个 pass 到底在做什么。

primitives 的对应关系大家应该很清晰了：fragment 物理上就是 register file；上层需要 inference 的逻辑任务和逻辑映射在 IR 里完成，让 TileLang 完成这个映射。memory hierarchy 不熟的话，这张图看明白了就全清楚了。layout inference 是我们三番五次讲的好 feature。

最后我们给了一些例子（没有放全），大家可以去参考 TileLang kernels（"TIKERNELS"）里的实践；我们之后也会评估这些代码质量够不够、单独整理一个东西发出来。对 shared memory 来说，我们这里以 padding 的方式避免 bank conflict，明天会介绍 swizzle 以及 tensor core。tile library primitives 是 TileLang 包好的 tile-level API；有一个下来再确认的点：`alloc_reducer`/`finalize_reducer` 到底被 lowering 成了什么，可以拿这个当小 demo 演示。

warp scheduler 是硬件特性：正因为这个硬件特性，我们才用 pipeline 的方式；occupancy 一定程度上可以反映性能——occupancy 高了之后 warp scheduler 的可选空间就大，但它不能直接代表性能。对这套硬件点了解清楚之后，很多上层的性能表现其实是硬件本身的反应。

最后对比的两个：pipeline 和 persistent。persistent 有更好的掌控力、能做 schedule control，但复杂度留给了程序员；任务特别规则的话，pipeline 就可以实现相当好的性能。case study 是 InstanceNorm backward（TileLang kernels 里的 case），看更多复杂的 mega kernel 可以直接学习那个实现、优化 mega kernel 会有更细致的心得，欢迎大家分享。

最后又提了一下 profiling：除了 NCU、NCS，还有 in-kernel profiler（IKET），也是强力推荐；配合 lowering trace 的方式去 play with it，看 kernel 内部到底是什么情况。以上就是我们今天的分享。

---

## 附：听辨存疑与说明

1. "TBM/TIRX、TBMFFI"等名称：字幕音译混乱，按上下文恢复为 TileLang 的 TIRX IR 与 TileLang FFI。
2. "in-kernel profiler（IKET）"：字幕记为 IKET/INCARNL，按上下文理解为 kernel 内 profiling 工具，名称未能完全确认。
3. "tile-notes"开源项目名：字幕记为 tier notes，按上下文恢复，未能完全确认。
4. "InstanceNorm backward"：字幕记为 INSTGRAM/ian grand，按上下文恢复为 TileLang kernels 中的 InstanceNorm backward 示例。
5. NCU 数据（stall 比例等）在本次演讲中未给出具体数值，转录只保留讲者明确提到的数字（如 16 warp×4 sub-partition=64、132 个 SM、256KB、75%～90% 有效带宽、num_stages 默认 3/最优约 4、8×12-bit→3×uint32）。
6. "MRCS 入门教程"：字幕听辨不清，可能指 IR/编译器入门教程，未强行补全。
