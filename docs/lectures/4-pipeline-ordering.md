# 4 Pipeline Ordering：Data Orchestration —— 完整讲课转录

## 视频信息

- 标题：北京大学未名超算队 × LCPU AI Infra Seminars 系列讲座：4-Pipeline Ordering: Data Orchestration
- 链接：https://www.bilibili.com/video/BV1NRuX6UEwJ
- UP 主：北京大学Linux俱乐部
- 主讲人：卢怡霏（Yifei Lu）
- 时长：1:33:00
- 讲座日期：2026 年 8 月 9 日
- 活动背景：2026 年暑期 Weiming HPC Training Camp × LCPU AI Infra Seminars 第五次活动录播

## 校订说明

本转录以 B 站 ai-zh 自动字幕为底稿，结合视频画面按语义校订。正文严格沿用原视频的讲述顺序，保留授课内容、例子、现场过渡和必要的重复说明；只删除纯语气词与无意义的口头重复，并恢复被字幕误识别的技术术语、公式和代码。小标题与时间范围是为方便阅读而补充，不改变讲者的论证次序。

字幕共 1656 条，覆盖全片（00:00:00.160—01:33:00.900），未重新运行 ASR。原始字幕与审阅媒体保存在本地 `course-lecture-builder/jobs/BV1NRuX6UEwJ/raw/`，不随网站发布。

---

## 一、开场：为什么需要 Data Orchestration（00:00—05:25）

今天这节课的主题是 Pipeline Ordering，主要讲 Data Orchestration。上一节课孙总已经给大家讲了 tensor core。Tensor core 的好处是能把各种 computation 的 throughput 拉得非常高，但这很容易带来一个矛盾：在 tensor core 需要下一块数据的时刻，我们不一定能保证数据已经待在正确的位置、已经完成传输，并且可以立即拿来计算。

所以这次讲的 Data Orchestration 主要考虑几个方面：

1. **Data movement**：数据从哪里搬到哪里；
2. **Storage**：数据在每一个阶段存在哪里；
3. **Synchronization**：搬数据的人和用数据的人，怎么知道数据已经到达，或者已经使用完毕；
4. **Scheduling / ordering**：每一个执行单元应该在什么时间做哪一份工作。

先过一下后面会用到的术语。CTA 基本就是 thread block，我们有时会把 block 叫作 CTA。另外我会经常说 producer 和 consumer。在这里，producer 指为下一阶段准备数据或任务的执行者；consumer 则是使用这些数据或任务的执行者。

### 一次 GEMM 的物理旅程

先看一个 GEMM 的 physical journey。数据要进入 tensor core 参与计算，首先会把 A、B 的若干 tile 从 HBM 搬到 shared memory；然后 warp 或 warp group 从 shared memory 取 operand，交给 tensor core。累加结果会放在 registers；在 SM100 之类的架构上，也可能放进 TMEM。最后，epilogue 再把结果写回 global memory。

这条路上有三个很直接的瓶颈。

第一，HBM latency 非常长。一次 global memory request 从发出到数据可用，要经历地址转换、cache 查询；如果 cache miss，还要访问更下层的 memory。

第二，register 和 shared memory 的容量都有限。我们不能无限 prefetch：每增加一个 stage，就会多占一份 shared memory；多保留几条独立计算链，也可能占用很多 registers。

第三，tensor core 确实太快了。如果 operand 晚几十到几百个 cycle 才到，那么好不容易获得的巨大 throughput，最后可能只是在空转。

对此有两个直观策略：reuse 和 overlap。

Reuse 主要解决“减少从远处搬数据”的问题。例如，一个 A tile 会被 N 方向的输出复用，一个 B tile 会被 M 方向的输出复用。可以先把它们放进 shared memory，让多个 thread 和 warp 重复读取，从而显著减少 HBM traffic。

但 reuse 只是减少搬运次数。即使总 traffic 变少了，每次真正需要去 HBM 取下一个 tile 时，latency 依然存在。所以还要考虑 overlap：等待期间去做其他事情。最直接的想法就是，当 tile $k+1$ 的 copy 仍在路上时，让 tensor core 计算 tile $k$。

### Little 定律

这件事也可以用 Little 定律来理解：

$$L=\lambda W$$

$W$ 是一项工作从发出到完成的平均时间，在这里就是 latency；$\lambda$ 是希望维持的完成速率，也就是 throughput；$L$ 是平均同时在途（in-flight）的工作数量。

如果 latency 很大，而我们仍希望 throughput 很大，就需要让足够多的工作同时处于 in-flight 状态。对于 GPU kernel，这些在途工作可以来自：同一 CTA 中许多尚未完成的 memory request、多个 pipeline stage、同一 warp 中彼此独立的 instructions，或者同一 SM 上已经 ready 的 warps。

## 二、从串行 Copy–Compute 到异步操作（05:25—08:49）

先看一个非常朴素的 baseline copy-compute loop。假设一个 block 合作搬运一个 tile，每个 thread 从 global memory 读取一个元素，再写入 shared memory。普通 load/store 会让数据显式经过 register：先执行 global load，把值放进 register；再执行 shared store。

```cuda
for (int k = 0; k < num_tiles; ++k) {
  copy_tile_to_smem(k);
  block.sync();
  compute_from_smem(k);
  block.sync();
}
```

这里要特别关注两个同步。

第一个 `block.sync()` 保护 producer 到 consumer 的 read-after-write dependency。在 consumer 读取 shared memory 之前，必须保证 producer 已经把对应数据写好。

第二个 `block.sync()` 保护后面会反复提到的 ownership。下一轮循环会覆盖同一块 shared memory，因此必须保证本轮所有 consumer 都已经读完，producer 才能重新使用这块 buffer 去搬其他数据。

由此，一块可复用的 shared-memory buffer 至少有两个状态转换：写入完成后从 empty 变成 full；消费完成后从 full 变回 empty。

这个 baseline 是正确的，但整个过程非常 serialized：搬运时不能计算，计算时下一块 copy 还没有发出，因此会暴露很多 latency。我们想做的就是前面提到的“copy $k+1$ 时 compute $k$”。

这就是 asynchronous operation 的基本想法：发起方发出一项操作以后，不必在同一条控制流上等它完成，可以继续执行与结果无依赖的工作，只在真正消费结果之前调用 wait。

```cuda
issue_async_copy();
do_independent_work();
wait_for_copy();
consume_result();
```

## 三、Cooperative Groups：明确谁参与协作（08:49—15:46）

在真正写 async copy 以前，还要解决一个问题：一个 tile 通常不是由一个 thread 搬完，而是 block 中很多 thread 合作搬运。因此必须说明一次 copy 由哪些 thread 调用、wait 又由哪些 thread 调用。

Cooperative Groups 很早就已经提出。为什么需要它？我们以前经常使用 `__syncthreads()`，但单看函数签名，看不出它要求整个 block 都参与。如果在分支中只让一半 threads 调用，那么进入函数的 threads 会停在这里等待同一 block 的其他 threads；其他 threads 永远不会进入，于是造成 deadlock 或 undefined behavior。

Cooperative Groups 所做的事，是让程序显式表达哪些 threads 组成一个 group，并在这个 group 上执行 synchronization、collective communication 或 partition。比如 `cg::this_thread_block()` 表示当前线程所属的整个 CTA。这个对象同时提供三类信息：

- participant set：哪些 threads 属于 group；
- 每个成员在 group 中的 rank；
- collective contract：执行 `block.sync()` 等 collective 时，group 成员必须共同参与。

这样，helper function 就可以把 group 写进接口。调用者能够明显看出，这个函数不是任意一个 thread 都能单独调用，而是接受一个 block-level cooperation contract。

当然，这不会自动杜绝 deadlock。如果拿到了 `thread_block`，却仍然只有一部分成员调用 `block.sync()`，程序依旧不正确。但如果只想让一个 32-thread tile 执行某个 collective，可以先让整个 parent group 调用 `tiled_partition`，再由目标 tile 进入自己的 collective。

```cuda
auto block = cg::this_thread_block();
auto tile32 = cg::tiled_partition<32>(block);

if (tile32.meta_group_rank() == 0) {
  warp_collective(tile32);
}
```

`tiled_partition` 本身也是 parent-group collective，不能把它放进“只有前 32 个 thread 进入”的 branch。整个 block 要一起 partition；随后 `meta_group_rank()` 在 tile 内是 uniform 的，第一组 32 个成员才会一起调用 warp collective。

常见 group scope 包括 warp-sized logical tile、CTA、thread block cluster 和 cooperative grid。Grid 比较特殊：grid sync 要求整个 kernel 通过 cooperative launch 启动，例如 `cudaLaunchCooperativeKernel`，而且 launched grid 必须满足同时驻留约束。如果一部分 CTA 已经驻留并停在 grid barrier，剩余 CTA 却因为资源不足无法启动，就会 deadlock；runtime 必须保证这种 grid cooperation 可行。

Cooperative Groups 用途很广，在 Data Orchestration 中也会反复出现。

### 用 group 表达 async copy

回到 async copy，可以按 Cooperative Groups 的写法表达：

```cuda
auto block = cg::this_thread_block();
cg::memcpy_async(block, smem_dst, gmem_src, bytes);
do_independent_work();
cg::wait(block);
consume(smem_dst);
```

这里有三个关键点：`block` 表示谁属于这次 collective；`memcpy_async` 发起 global memory 到 shared memory 的异步搬运；`wait(block)` 让 group 确认此前发出的 collective copies 已经完成。

现代 `memcpy_async` 同时提供 non-group 和 group overload。没有 group 参数时，copy 由当前 calling thread issue；多个 threads 执行这条语句，就各自 issue 一次 copy。带 group 参数时，它是 cooperative overload，group 成员共同 issue 这一段 copy；所有成员必须参与，而且要对 collective 参数达成一致。

Async 的正确性并不能仅由 wait 保证。无论使用哪一种 API，都还要保证三件事：

1. **Addressing/range**：谁搬哪一段，pointer 是否 valid，有没有越界，source 与 destination 是否重叠，alignment promise 是否真实；
2. **Completion**：consumer 在哪个 token、group、pipeline stage 或 barrier phase 上等待；
3. **Reuse**：上一轮 consumer 读完之前，下一轮 producer 会不会覆盖同一个 shared-memory stage。

Wait 只说明“这个 completion mechanism 所追踪的异步工作已经完成”。它不会替你保证 source/destination 正确，也不会保证你等的是正确 stage。因此，写了 wait 不等于程序就一定正确。

## 四、SM80：`cp.async` 与 Pipeline（15:46—22:08）

下面按照不同代架构分别来看。SM80 引入的重要能力，是把普通 CUDA path 上两条数据移动合在一起。原本的 `LDG` 把 global memory 搬到 thread register，`STS` 再从 register 搬到 shared memory；SM80 提供了 `LDGSTS`，在 PTX 层面对应 `cp.async`，把 global load 和 shared store 拼成一条 global-to-shared 路径。

这样 payload 不一定要在 register 中显式转一圈。收益很明显：减少一次显式 register transit，降低相关 instruction overhead；发起搬运以后，warp 也可以继续执行 independent work。这为后面构造 multi-stage global-to-shared pipeline 提供了硬件基础。

### Alignment 条件

`LDGSTS` 路径有一定条件。硬件搬运按照固定 transaction granularity 工作；具体细节会随 API 和架构变化，但 source 与 destination 必须满足要求的 alignment，copy size 也必须能由对应 transfer granularity 正确表达。

实践中，4-byte alignment 可以看作最低的基础条件；16-byte alignment 更容易稳定进入高效 async path，因此更加推荐。不能把 alignment promise 写得比实际地址更强。

### Warp entanglement

另一个细节叫 warp entanglement。某些与搬运、提交和消费有关的状态是在 warp 内共享的。每个 thread 可以看到自己的 sequence，但整个 batch sequence 会受到 whole-warp commit 行为影响。

如果一个 warp 在 divergent control flow 中提交不同数量的 async-copy batches，warp-wide sequence 可能前进得比某些 thread 自己认为的更远。后果包括：一些 threads 等待了比自己真正需要的更多 batches，barrier 的 arrive-on 被更新更多次，原本希望由 overlap 节省的时间又被多余等待吃掉。

因此 CUDA guide 建议，commit 和 arrive-on 尽可能在 converged warp 上执行，也就是相关 active lanes 处于同一条控制路径、执行同一个 protocol step。

### N-stage FIFO 与 ownership cycle

现在来看 pipeline。它可以简单理解为一个 N-stage、first-in-first-out 的队列。只有一个 shared-memory buffer 时，copy 与 compute 很难重叠，因为下一次 copy 会立刻覆盖当前 compute 使用的数据。准备两个 stages，就是 double buffering。

一个 CUDA pipeline 存在清晰的 ownership cycle：

1. `producer_acquire`：producer 取得一个 empty stage 的写入权；
2. producer issue 自己的 async copies；
3. `producer_commit`：提交这一批生产工作；
4. `consumer_wait`：等待最老的 full stage 可用；
5. consumer 对该 stage 执行 compute/consume；
6. `consumer_release`：把 stage 重新标记为 empty，还给 producer。

Producer 只能写它 acquire 到的 empty stage；consumer 只能读 wait 完成后的 full stage。`consumer_release` 扮演的角色，与 baseline 中第二个 `block.sync()` 很相似，都是保护 buffer reuse。

Double buffering 可以分成三个阶段：

- **Prologue**：计算开始前先搬入第一个 tile，此时没有旧 tile 可算；
- **Steady state**：每轮同时为 future tile 发 copy，并计算 current tile；
- **Drain**：最后一块 future tile 已经发出，还要等待它完成，执行最后的 wait、compute 和 release。

```cpp
prime(stage[0]);

for (int k = 0; k < K_tiles - 1; ++k) {
  acquire_empty(stage_for(k + 1));
  issue_copy(stage_for(k + 1), tile(k + 1));
  commit();

  wait_until_full(stage_for(k));
  mma(stage_for(k));
  release(stage_for(k));
}

drain_the_final_stage();
```

## 五、为什么 Double Buffering 仍可能隐藏不了延迟（22:08—25:58）

有了 double buffering，并不代表所有 latency 都会自动消失。先把 warp 分为三类：

- **Active warp**：已经 resident 在某个 SM 上、尚未退出；active 不代表现在能运行；
- **Eligible warp**：active warp 中，下一条 instruction 已经准备好，可以被 scheduler 选中发射；
- **Issued warp**：真正被选中，并已经发射 instruction 的 warp。

一个 warp 可以 active 但不 eligible。例如，它的下一条 instruction 依赖一个尚未从 global memory 返回的 register，或者它正在等 barrier。

Warp stall 本身很正常。GPU 的设计就是在一个 warp 等待时寻找另一个 eligible warp。真正损失 throughput 的情况，是 scheduler 有 issue opportunity，却找不到足够的 eligible warps 或 instructions。

Double buffering 常见的失败原因有三个。

第一，copy 发得不够早。如果 current compute 的时间短于 future copy latency，走到 wait 时数据还没到，仍会等待。

第二，所有 warps 在相同时刻走到同一个 wait。即使 CTA 中有很多 warps，只要它们处于同一个 pipeline phase，也可能一起失去 eligibility。

第三，资源限制使可替换的工作太少。更多 stages 占用 shared memory，更多 unrolling 占用 registers；每个 CTA 都很臃肿时，一个 SM 只能 resident 更少的 CTAs 或 warps。

### Occupancy 不是利用率

Occupancy 通常指一个 SM 上实际 resident 的 active-warp 数量，相对于架构允许的 maximum resident warps。比如架构每个 SM 最多容纳 64 个 resident warps，而资源用量只允许放 32 个，那么理论 occupancy 就是 50%。

限制 occupancy 的因素很多，包括 threads per block、registers per thread、shared memory per block，以及架构本身对 block 或 warp 数量的上限。

但 occupancy 不等同于利用率，也不等同于 warp-ready 比例。即使达到 100% occupancy，如果所有 warps 都等待同一条长依赖，执行单元仍会空闲。反过来，较低 occupancy 的 kernel 如果 overlap 做得好，也可以跑满关键 pipeline。所以 occupancy 更像 latency-hiding capacity 的指标，而不是最终性能答案。

## 六、Pointer Chasing 实验：依赖链如何造成停顿（25:58—33:45）

下面是一个比较刻意、并不严谨的小实验，目的是让大家理解 dependency scoreboard、latency hiding，并顺便看一下 Nsight Compute。

实验使用大家的 RTX 5090 节点，一个 GPU，launch configuration 是 170 blocks、每个 block 32 threads，两种 kernel 使用相同 iteration 数。主要区别是每个 thread 维护一条 pointer chain，还是两条互不依赖的 pointer chains。

### 单链

```cuda
unsigned p = tid & mask;
#pragma unroll 1
for (int i = 0; i < iters; ++i)
  p = next[p];
out[tid] = p;
```

可以把 `next` 想成一个很大的数组，数组元素仍是下一次访问的数组下标。例如当前 `p=17`，这次读取 `next[17]`；若返回 932，只有拿到这个结果后，下一轮才知道应该读 `next[932]`。

时间线上，要先计算 `next[p]` 的地址，发出 global load，等待 load 返回新的 `p`，再用它计算下一次地址并发出下一次 load。下一轮地址依赖本轮数据，无法提前知道。

在 Nsight Compute 中看 SASS，热循环最关键的是 `IMAD.WIDE` 与 `LDG.E`。前者读取保存下标的 register，把数组下标转换为 byte address；load 返回后，新的数组元素又写回这个 register。循环回到开头，下一次地址计算继续读取它，因此构成封闭 dependency chain。

这里会遇到 scoreboard。Scoreboard 保证 consumer 不会读取尚未 ready 的 register：硬件跟踪未完成的 dependency。当某条 in-flight `LDG` 将来要写入一个 register、但当前还没 ready，而 warp 下一条 instruction 已经要读取它时，scheduler 会发现 operand 未就绪，该 warp 就不能成为 eligible warp。Nsight Compute 会把这种等待报告为 Long Scoreboard stall。

在这个实验中，one-chain 平均每条 issued instruction 对应约 199.85 个 warp cycles，其中约 193.1 cycles 是 Long Scoreboard，占 96.6%。这个指标并不严格等于“每条 global load 固定 latency 为 193.1 cycles”，它是该 workload 按 issued instructions 归一化的 warp-stall metric，主要用来确认等待类型。

Scheduler statistics 也能看到前面三个概念。实验平均只有一个 active warp，而 eligible warp 只有约 0.01。Warp 并没有消失，只是停在需要该 register 的 consumer instruction 上。No Eligible 达到约 99.5%，表示 scheduler 在几乎整个采样周期里都找不到能够 issue 下一条 instruction 的 warp。这才是 latency 没被隐藏而导致 throughput loss 的直接表现。

### 两条独立链

Two-chain 情况会好很多，因为每个 thread 有两个独立起点。`p0` 的下一步依赖 `p0`，`p1` 的下一步依赖 `p1`，但二者互不依赖。

```cuda
unsigned p0 = tid & mask;
unsigned p1 = (p0 + ((mask + 1) >> 1)) & mask;

#pragma unroll 1
for (int i = 0; i < iters; ++i) {
  p0 = next[p0];
  p1 = next[p1];
}
out[tid] = p0 ^ p1;
```

SASS 中也会出现两组地址计算与 `LDG.E` dependency chain。这样发生了两个层次的变化：同一个 warp 的 instruction stream 多出一条 independent chain，这叫 instruction-level parallelism（ILP）；两条 independent loads 也可以同时 outstanding，这叫 memory-level parallelism（MLP）。前者描述 instruction dependency graph，后者描述 memory system 中并发在途的 requests。

从时间看，one-chain 约 2.10 ms；two-chain 约 2.81 ms，但它完成了约两倍工作，因此单位工作吞吐接近提升 50%。

除了 ILP 和 MLP，还可以用 thread-level parallelism（TLP）隐藏延迟：增加 resident warps，让一个 warp 等待时由另一个 warp issue。

由这个实验可以总结几种调度思路：

- 在同一 warp 中增加 independent chains，提高 ILP，并由此增加 outstanding loads 和 MLP；
- 提前发出 future tile 的 async copy，让 future transfer 与 current compute 同时在途，增加跨 tile 的 MLP；
- 增加 resident warps，用 TLP 提供替代工作；
- 增加 pipeline stages，给 copy 更长的提前量，但会消耗更多 shared memory 和状态。

## 七、SM80 小结：四个同步问题（33:45—36:40）

走到这里，可以把 Pipeline Ordering 中需要同时考虑的同步问题归纳为四个方面。

第一是 **completion**：异步操作是否完成，例如 future slot 的 copy 是否结束。

第二是 **visibility**：即使 producer 自己认为已经完成，consumer 是否看得到结果。这与 memory scope 和 synchronization semantics 有关。

第三是 **ordering**：如果数据由不同 execution agent 或 memory proxy 访问，它们之间的先后关系是否已经建立。后面讲 TMA 时会看到，普通 thread ordering 不一定覆盖所有 async path。

第四是 **ownership**：谁现在可以使用这片 storage；consumer 读完以前，producer 能不能覆盖它。最简单的表达就是 full/empty stage protocol。

Stages 显然不是越多越好。更多 stages 意味着更多 shared memory，可能降低 occupancy，也需要更多 control state。

SM80 的 takeaway 是：它让 global-to-shared 的 async copy 更直接，减少 register 作为显式中间站，为 copy-compute overlap 提供硬件基础。但程序员仍要设计 participant group、每个 thread 的 address assignment、alignment、shared-memory layout、stage count 和 ownership，还要平衡 shared memory、registers 与 occupancy，并决定操作要提前多久 issue、期间是否真有 independent work 可以 overlap。

因此一个自然的下一代优化方向是：不要再让很多 threads 各自算地址、发许多小 copy，而让一个指定 issuer 描述整块 tensor transfer，把更多 producer 工作交给硬件。这就是 Hopper 所做的事情。

## 八、SM90：TMA 与 Tensor Map（36:40—40:32）

SM80 的典型做法，是每个 thread 发起自己的 small copy。Hopper 引入 Tensor Memory Accelerator（TMA），允许一个 elected thread 发起一块多维 tensor 的 bulk asynchronous transfer。硬件依据预先编码的 descriptor 生成各元素地址并完成搬运。也就是说，由 TMA 发 instruction，而不是每个 thread 发一大堆指令。

### Tensor map 是 transfer contract

TMA 不希望每次 issue 时都由 CUDA instructions 重新计算整个 tile 的地址，所以需要 tensor map，也就是 descriptor。它描述的内容包括：

- global-memory base pointer；
- tensor rank、各维 extent 与 global byte strides；
- transfer box/tile dimensions；
- element strides；
- shared-memory interleave 或 swizzle；
- L2 promotion policy；
- out-of-bounds fill policy。

Host 侧通常通过 driver API，例如 `CUtensorMapEncodeTiled`，构造 descriptor。Kernel 发 TMA 时主要提供 descriptor、当前 tensor coordinates、destination 和 completion object。

所以可以把 tensor map 理解为 transfer contract：从这个多维 source 出发，按这些 strides 解释内存，取出指定 box，再按照指定 shared-memory layout 放下。

L2 promotion 可以选择更大的 fetch/promotion granularity，例如 64 B、128 B、256 B。直观地说，它告诉 memory system：访问目标区域时，可以尝试把相邻的更大 segment 提前搬入，供后续访问复用。

Out-of-bounds fill 则服务于边界 tile。Box 的一部分可能超出 tensor extent；传统写法要由 threads 逐元素写 predicate，而 tensor map 的 OOB fill 可以让 TMA 对越界位置填入定义值，通常是 zero fill，从而简化边界处理。

到这里，tensor map 回答的主要还是“怎么搬数据”。TMA issue 后，异步 agent 在后台工作，CTA threads 继续执行。新的问题是：一个 elected thread 发出 box transfer 后，其他 consumer threads 用什么对象确认整块 transaction 已经真正到达 shared memory？

`__syncthreads()` 做不到这一点。它只能说明 threads 到达同一个 barrier，不能自动包含“一项独立的 TMA transaction 已完成”这个条件。因此需要 `mbarrier`。

## 九、`mbarrier`：不仅等线程，还要等事务（40:32—44:09）

为什么 SM80 的讲法里没有突出这个问题？`cp.async` 主要把一批 small copies 组织成 async groups 或 batches，再用 sequence-based wait 表达“此前的 groups 已完成”。TMA 的需求不同：它是一个由 elected thread issue 的 box transaction，producer 和 consumer 很可能来自不同 warp groups，同一 CTA 还会循环复用许多 shared-memory stages。

`mbarrier` 的 completion 有两个条件：

1. **Thread arrival**：期望的 threads 都已到达；
2. **Async transaction completion**：与当前 phase 绑定的 async transactions 满足完成条件，例如承诺的 transaction bytes 全部到达。

只有两者同时满足，barrier 才切换 phase，让程序从“搬数据”进入“消费数据”。高层 CUDA barrier 的示意写法是：

```cuda
using barrier = cuda::barrier<cuda::thread_scope_block>;
__shared__ barrier bar;

if (threadIdx.x == 0)
  init(&bar, blockDim.x);
__syncthreads();

if (is_elected()) {
  cuda::memcpy_async(
      smem, gmem,
      cuda::aligned_size_t<16>(bytes),
      bar);
}

auto token = bar.arrive();
bar.wait(std::move(token));
consume(smem);
```

### 为什么必须跟踪 phase

同一个 barrier object 会在循环中反复使用。每完成一次，phase 就翻转。在底层代码中，经常用 parity bit 或 token 区分究竟等待哪一轮。如果不维护 phase，就可能错误地把上一轮完成当成本轮完成。

因此每个 stage 要绑定自己的 barrier state 或 phase。不能只写一个全局布尔值 `ready`，因为它无法可靠区分 generation，也不能完整表达 transaction completion。

高层 API 中，arrival token 不是简单的 ready/not-ready 布尔值。它与调用 `arrive()` 时所在的 barrier phase 绑定。例如 thread 在 phase 0 调用 `arrive()`，随后其他 participants 让 barrier 完成并进入 phase 1，那么 `wait(token_for_phase_0)` 就能通过。

底层 PTX 可以显式跟踪 parity：

```cpp
auto native = cuda::device::barrier_native_handle(bar);
int parity = 0;

for (...) {
  (void)cuda::ptx::mbarrier_arrive(native);
  while (!cuda::ptx::mbarrier_try_wait_parity(native, parity)) {}
  parity ^= 1;
}
```

Parity 只有一个 bit，所以 phase 0、2、4 的 parity 都是 0，phase 1、3、5 都是 1。它不是无限递增的 generation number，只是 0、1 交替。

## 十、Generic Proxy、Async Proxy 与完整 TMA 往返（44:09—48:14）

下面是 async proxy 与 generic proxy。普通 CUDA instructions 执行 load/store，与 TMA hardware 在后台搬数据，并不是同一个 execution agent，也不一定通过同一种 memory-access mechanism。NVIDIA memory model 用 proxy 区分这些访问路径。

默认情况下，两条路径可以访问同一个 shared-memory address；proxy 讨论的不是“地址能否访问”，而是 ordering。通过不同机制发出的 accesses，不能自动假设会按照 source code 的书写顺序相互可见，因此需要 proxy fence 把路径连接起来。

可以把完整的 TMA 往返分为三部分。

### 1. Global memory 到 shared memory：等待数据 ready

`cuda::memcpy_async` 把 copy completion 绑定到当前 barrier phase。Required thread arrivals 与 transaction bytes 都满足后，barrier 翻转 phase，`bar.wait()` 越过 completion boundary。此后 shared memory 中的数据才能被等待的 threads 安全读取。

### 2. Generic 写入交给 async proxy 读取：建立可见性

接下来，普通 threads 通过 generic proxy 修改了 shared memory，而后续 TMA 要通过 async proxy 读取这些内容。每一个参与写入的 thread 都要执行 `fence_proxy_async`，把该 thread 在 fence 之前的 shared-memory generic accesses 排到后续 async-proxy accesses 前面，使稍后的 TMA read 必须观察到这些 ordinary stores。

```cuda
cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
__syncthreads();
```

Fence 本身不是 rendezvous。一个 thread 执行 fence，并不表示其他 writers 都写完了，所以还需要 `__syncthreads()` 保证所有 writers 到达。这里 fence 与 barrier 各自解决不同问题，不能相互替代。

### 3. TMA 读完 shared memory：才能复用 source buffer

TMA 从 shared memory 向 global memory 发出 bulk async copy 后，还要确认它已经完成对 shared-memory source 的读取，buffer 才可复用。这里使用的 completion mechanism 不是前面的 shared `mbarrier`，而是 issuing thread 的 bulk async group：

```cuda
if (is_elected()) {
  cuda::ptx::cp_async_bulk(/* smem -> gmem */);
  cuda::ptx::cp_async_bulk_commit_group();
  cuda::ptx::cp_async_bulk_wait_group_read(
      cuda::ptx::n32_t<0>());
}
```

Commit-group 把此前 bulk operations 提交成一组；wait-group 等待这些 operations 完成对 source shared memory 的读取。函数名中的 `read` 正是在保护 ownership：只有 read completion 以后，这块 source shared memory 才能安全复用。

至此，才得到一个相对完整的 TMA pipeline：先用 `mbarrier` 建立 global-to-shared completion，再用 proxy fence 加 CTA rendezvous 建立 generic-to-async visibility，最后用 bulk read completion 保护 shared-memory source reuse。

## 十一、Swizzle 与 Bank Conflict（48:14—50:12）

Shared memory 被划分为很多 banks，一个 warp 可以同时访问多个不同 banks，并行服务这些请求。如果许多 lanes 在同一时刻访问同一个 bank 的不同地址，请求很可能被拆成多次服务，这就是 bank conflict。

TMA swizzle 所做的，是改变 tile 落到 shared memory 时，logical coordinates 到 physical shared address 的映射。目标是让 consumer warp 或 warp group 的变形访问更均匀地分布到各个 banks。

Swizzle 也带来约束。硬件按固定 pattern 重排地址，因此 swizzle mode 会约束 tensor box/row byte width、global base 或 stride alignment、shared-memory swizzle base alignment 和 granularity。这里的 access unit 是固定的 16 bytes。实际编程时需要查对应架构文档中的表格，不能只看逻辑 tensor shape。

## 十二、Warp Specialization（50:12—53:00）

有了 TMA，一个或少数 elected threads 就能发起 bulk transfer，让所有 warps 轮流执行同一份 copy-control code 已经没有必要。Warp specialization 的核心想法，是把同一 CTA 中的 warp groups 分成长驻、角色不同的 agents。

在典型 Hopper GEMM 中：

- producer warp group 负责 TMA issue，以及 empty/full stage 的转换；
- consumer warp groups 等待 full stage，发 WGMMA，维护 accumulators，并在完成后 release stage；
- 某些 schedule 还会指定一个 consumer 承担 epilogue。

分角色有三个明显好处。

第一，producer control flow 与 consumer math control flow 分开，减少每个 warp 都执行那些与自身角色无关的 instructions。

第二，不同角色可以使用不同 register budget。Producer 不需要大量 accumulators；consumer 也不需要保存所有 address-generation state。

第三，producer 可以持续填满 future stages，而 consumers 按自己的 math-pipeline 节奏消费。

但 ownership cycle 并没有变化：producer acquire empty stage，elected issuer 发 TMA，`mbarrier` 等待 stage full，WGMMA 使用 operands，consumer release，producer 再次获得 empty stage。Warp specialization 改变的主要是“每个动作由谁执行”，动作本身和依赖关系仍然一样。

## 十三、SM100：TMEM 与新的 Orchestration 责任（53:00—55:22）

接下来是 Blackwell SM100。Hopper 及以前，accumulator 长时间放在 consumer threads 的 registers 中，会产生资源压力。输出 tile 越大，accumulator 越多，consumer warp group 使用的 registers 越多；register pressure 又会限制 occupancy，让其他安排更困难。

TMEM 为 tensor-core accumulator 提供专门的存储空间。`tcgen05` 数据流的要点是：同一条 MMA 由一个 elected thread 发起；A operand 可以来自规定的 shared memory 或 TMEM source，B operand 通常来自规定的 shared memory；accumulator destination 放在 TMEM。有时还可以使用 two-CTA MMA，引入 CTA-pair scope 的 lifetime 与 synchronization。

从 orchestration responsibility 的演变来看：

- SM80 主要管理 copy-stage lifetime；
- SM90 增加 bulk transfer、`mbarrier` phase 与 warp roles；
- SM100 又增加 TMEM allocation/deallocation lifetime、MMA 对 TMEM accumulator 的读写顺序、LDTM 与 epilogue 的 handoff；使用 two-CTA mode 时，还要处理 CTA pair 的 cooperation 与 termination。

这些内容上一讲已经详细介绍，这里主要是把它们放回 Data Orchestration 的视角中。

## 十四、MoE 为什么带来不规则 GEMM（55:22—57:47）

一个设计得很好的 tile，不一定构成一个很好的 workload，因为现实中有许多不同使用场景。Mixture of Experts 就是典型例子。

普通 feed-forward layer 会让每个 token 经过同一组 weights；MoE layer 准备许多 expert networks，再由 router 为每个 token 选择少数 experts。执行过程大致是：router 读取 tokens、选择 experts；tokens 按 expert 分组；每个 expert 对分给自己的 tokens 执行 GEMM；最后按原 token 顺序合并结果。

于是一个 batch 中会出现许多个 $M\times N\times K$ problem，特别是 $M$ 会随 routing 动态变化。即使每个单独 GEMM 都使用了 TMA 和 WGMMA，整体计算仍可能很慢。

例如，静态给每个 SM 分配相同数量的 tiles，并不能保证时间相同。不同 problem 的边界、K、epilogue 与 cache behavior 不完全一样。先完成的 SM 会空闲，少数 SM 还在处理困难工作，形成 tail imbalance。

有两个主要策略：

1. **Grouped GEMM**：把多个 GEMM problems 放进同一个 kernel；
2. **Persistent kernel**：让有限数量的 resident workers 反复领取 tiles，而不是做完一个 tile 就退出。

先看 persistent kernel。

## 十五、Persistent Kernel：让 Worker 持续工作（57:47—64:34）

Persistent kernel 的核心是 keep workers alive。传统 tiled GEMM 通常为每个 output tile 启动一个 CTA。每个 CTA 走完整个 K loop，完成自己的 epilogue 后退出。Logical grid 有多少 output tiles，就有多少 CTAs 等待 hardware scheduler 分批运行。

Persistent kernel 则启动一个有界的 worker grid，按 occupancy policy 选择 CTA 数量。每个 worker CTA 完成一个 output tile 后不会立即退出，而是向 software scheduler 请求下一个 tile；没有剩余工作时才结束。

这样做有几个收益。

第一，固定驻留的 workers 可以复用一部分 setup state，例如 role partition、descriptors、scheduler state 和 barriers。

第二，workers 动态领取工作，有机会缓解前面提到的 tail imbalance。

第三，许多小 GEMM 可以合并到一次 launch 中，减少 launch overhead。

第四，同一 worker 内部可以反复运行 TMA→MMA 的 multi-stage pipeline。

理解上可以把它看成两层 pipeline：inner pipeline 是一个 output tile 内，沿 K 维多个 stages 的 data movement 与 compute overlap；outer pipeline 则包在外面，让 worker 在许多 output tiles 或 problems 之间不断领取、切换并复用资源。

### Persistent outer loop

用高度抽象的伪代码表达：

```cpp
WorkerState state = initialize_once();

while (scheduler.next_tile(work)) {
  bind_problem_and_coordinates(state, work);
  compute_one_output_tile(state, work);
  finish_epilogue_and_release(state, work);
}

drain_and_exit(state);
```

`initialize_once` 建立 warp roles、barriers、pipeline objects 等 per-worker state。

`next_tile` 请求 work item，返回 problem id 与 coordinates；没有剩余工作时返回 false。

`bind_problem_and_coordinates` 根据 problem id 选择 A/B/C/D pointers、strides、M/N/K 等参数，并把当前 tile coordinates 绑定到 descriptors。

`compute_one_output_tile` 运行 K loop，也就是完整 inner pipeline。

`finish_epilogue_and_release` 完成 D store，并确保这个 tile 的 temporary state 已经安全释放。

最后 `drain_and_exit` 处理仍在 in-flight 的 async operations，并完成必要的 termination protocol。

### 三种 lifetime

为了避免状态混乱，可以区分三种 lifetime。

**Per-worker state** 跨多个 tiles 长期存在，包括 role assignment、scheduler state 和 barrier objects。

**Per-stage state** 在一个 worker 内反复复用，包括 shared-memory slot、full/empty phase 与 TMA completion token。

**Per-tile state** 每拿到新 work 都要更新，包括 problem id、M/N coordinates、base pointers、valid bounds 和 epilogue metadata。

### Persistence 的代价

Persistent workers 活得很久，也会长期持有 shared memory 和 registers。这可能让大部分资源被少数 workers 长时间占据。

Termination 也会更复杂，因为必须正确 drain 所有 roles 和 stages。长期驻留还会减少与其他 GPU workloads coexist 的机会；如果 static mapping 做错，仍可能出现严重负载不均衡。所以 persistent kernel 不一定总是更好，需要按 workload 使用。

## 十六、Persistent Ping-Pong（64:34—66:12）

另一种 persistent kernel 叫 persistent ping-pong，名字很形象。它通常有一个 producer 和两个 consumer warp groups。当一个 consumer warp group 为 tile A 做 epilogue 时，另一个 consumer warp group 开始 tile B 的 MMA；之后二者交换。

这样可以让一个 tile 的 epilogue 与另一个 tile 的 main loop overlap。收益包括 MMA/epilogue overlap，producer setup 与 persistent scheduling state 跨 tiles 复用，并且对某些 shapes 改善 pipeline utilization。

代价是更多 live state：同时存活的 tile state 和 accumulator state 都会增加。Termination 更困难，必须确认两个 consumers 与 producer 都被正确 drain。资源占用增加后，也可能限制 coexistence。

## 十七、Cluster Launch Control（66:12—70:03）

Persistent kernel 有一个基本 trade-off。

一种做法是按 logical work item 启动足够多 blocks，让硬件正常调度。Load balance 和与其他 workload coexist 比较自然，但每个 block 通常只处理固定工作，无法充分复用 setup。

另一种做法是只启动固定数量的 persistent workers，让它们循环处理所有 work。这样 per-work setup overhead 较低，可以复用 state；但 worker 数在 launch 时已经决定。如果运行时部分 SM 被其他 kernel 占用，或 workers 启动节奏不同，static mapping 仍会形成尾部不平衡；长期 resident workers 也可能减少其他工作获得调度的机会。

Blackwell 提出了 Cluster Launch Control（CLC）。官方表格比较三种 policy：fixed work per block、fixed number of blocks 和 CLC。CLC 的设计目标，是尝试结合前两者的优点。

Fixed work per block 容易 load balance 和 coexist，但不强调 persistent state reuse；fixed-number persistent blocks 的 setup overhead 很低，但 load balance 和 coexistence 风险更高。CLC 让 resident blocks cancel 或 steal 尚未开始的 logical work，从而既复用自身，又允许未开始的工作动态转移。

它的逻辑是：worker 完成自己的 block index 后，请求取消一个尚未启动的 logical work。请求成功，就得到被取消 work 的 coordinate，当前 worker 转去计算它，然后再次请求；请求失败，表示没有可以偷来的工作，于是结束。

### CLC request 自己也是 async producer

普通函数调用结束后，结果可能立即回到 register；CLC request 显然不是这样。它也可以被看作 pipeline producer，只不过生产的不是 A/B tile，而是下一份 work 的 coordinate。

因此可以故意把 request 放到当前 tile 的 compute 前面，让 current computation 隐藏 work-acquisition latency：

```cpp
while (have_work) {
  request_next_work_async();
  compute_current_tile();
  have_work = wait_and_receive_next_work();
}
```

这是把同一套 async/pipeline 思路应用到调度本身。

## 十八、Grouped GEMM：把多个问题拼成逻辑 Tile 空间（70:03—76:30）

回到 MoE。一次 inference 里不是只有一个规则的大 GEMM，而是许多形状不同的小 GEMM。不同 problem 可能有不同 M/N/K、base pointers、strides 和 epilogue metadata。如果每个小 GEMM 都单独 launch，一个很小的 GEMM 可能没有足够多 output tiles 填满 GPU，反复 launch 也会增加调度开销。

Grouped GEMM 把这些 GEMM 组成 problem list，交给同一个 kernel，让一组 persistent workers 持续领取不同 groups 的 output tiles。

第 $g$ 个 GEMM 的 output matrix 在 $M\times N$ 平面切成：

$$
n_m^{(g)}=\left\lceil\frac{M_g}{B_M}\right\rceil,
\qquad
n_n^{(g)}=\left\lceil\frac{N_g}{B_N}\right\rceil
$$

它的 output tile 数为：

$$n_{tiles}^{(g)}=n_m^{(g)}n_n^{(g)}$$

Grouped scheduler 不再把这些 tile sets 看成彼此无关的 launches，而是把 GEMM 0、GEMM 1、GEMM 2 的 tiles 首尾相接，组成一个 global logical tile-id space。

讲义图中每个小矩形是一个 output tile，矩形里的数字是负责它的 persistent block id，而不是 global tile id。因此相同数字会在不同位置重复，表示同一个 block 做完一个 tile 后继续做另一块，甚至可能跨到下一个 GEMM problem。

### Static round-robin

固定 workers 可以按整个 persistent grid 的大小跨步遍历 logical tile ids。这里 `NUM_SM` 表示 program instances，也就是 persistent worker-grid 大小；并不是说 program 0 永远绑定 physical SM 0，实际 placement 仍由 GPU hardware scheduler 决定。

每个 worker 从自己的 program id 开始，按 grid size 跨步：

```python
tile_idx = tl.program_id(0)
while tile_idx < total_tiles:
    process(tile_idx)
    tile_idx += NUM_SM
```

例如 `NUM_SM=4` 时，program 0 处理 0、4、8、12；program 1 处理 1、5、9、13，其他 program 类似。这叫 static round-robin schedule。Worker 不需要访问全局 queue，只要知道初始 id 与 stride，就能独立算出自己的序列。

示例代码中还有两层 loop：外层 `for g in range(group_size)` 扫描 group ranges；内层 `while tile_idx < end` 表示该 worker 下一个 strided id 仍属于当前 group。

Round-robin 往往能让 workers 获得相近数量的 tiles，但 tile 数量相同不代表运行时间相同。边界 tile 需要更多 predicate，data conversion 和 epilogue 复杂度不同，cache locality 也可能不同。Static schedule 仍容易出现 tail：一部分 workers 已经结束，另一些还在处理昂贵 tiles。带 work queue 的动态 acquisition，或前面讲的 CLC，可以缓解这个问题。

### 从 global tile id 映射到具体 problem

Worker 得到 global tile index 后，还必须知道它属于哪个 group，以及在该 group 中对应哪个 local tile、tile M 和 tile N。

做法是为各 group 建立 prefix ranges。假设三个 groups 分别有 6、2、6 个 output tiles，那么区间是 $[0,6)$、$[6,8)$、$[8,14)$。若当前 global tile index 是 9，它不在前两个区间，而落在 group 2 的 $[8,14)$ 中。减去区间起点得到 local tile id，再按该 group 的 $n_n$ 拆出 tile-M 与 tile-N 坐标。

## 十九、跨 Group 边界时哪些状态必须更新（76:30—78:33）

把前面几部分串起来，一个 persistent worker 的外层路径是：取得下一个 global tile id；找到 group id；算出 tile-M、tile-N；绑定当前 group 的 metadata；执行一个 output tile 的 inner pipeline；epilogue 把结果写回 `D[group]`；然后继续下一项工作。

下一块工作可能属于不同 group。Kernel 不会重新 launch，worker 也不会重新创建，但它当前代表的 logical problem 已经变了。因此要区分两类 state。

可以继续长期存在的，是 producer/consumer roles 和整个 pipeline 的 stage protocol。

必须随 group 更新的，包括 group id、A/B/C/D base pointers、runtime strides、descriptors、predicates 和 epilogue destination。跨 group 不是只换一个 base pointer；上一组的 descriptor 与边界条件也必须重新解释，这正是 grouped persistent kernel 容易出错的地方。

## 二十、其他加速器如何安排数据（78:33—79:15）

最后进入一点“娱乐时间”。GPU 是一种 accelerator，前面讲的是 NVIDIA GPU 用哪些办法加速 Data Orchestration。Intel 和 AMD 的 GPU 也有类似思路，只是名字不同，所以这里选择三种组织方式更不一样的 accelerator：TPU、Graphcore IPU 和 Groq TSP。

## 二十一、TPU：把数据移动固化为空间波（79:15—82:36）

Google 把 Cloud TPU 描述为面向 machine-learning matrix operations 的 application-specific processor。这里的 application-specific，指硬件资源重点服务于 matrix workload，而不像 CPU 那样追求任意通用程序。

官方动画展示了数据怎样流动：host 把输入交给 TPU 队列叫 infeed；TPU 把计算结果还给 host 叫 outfeed；HBM 是设备侧的大容量 high-bandwidth memory；MXU 是 Matrix Multiplication Unit，内部包含执行密集矩阵乘加的 systolic arrays。

Systolic array 由规则排列、彼此连接的 multiply-accumulate cells 组成。数据从 array 边缘注入；每个 cell 用当前到达的 operands 做 multiply and accumulate，再把仍需使用的数据直接传给相邻 cell。

这个执行过程也有三个阶段。

第一是 fill。数据像 wavefront 一样进入 array，最开始只有靠近入口的一部分 cells 在做有效工作。

第二是 steady state。数据按节拍在 cells 之间传播，大量 cells 同时计算。

第三是 drain。最后一批输入已经注入，但部分 partial results 还要继续完成，最后才离开 array。

这里的重点是空间 reuse。一个从 HBM 进入 MXU 的 operand 不需要由每个 cell 各自从 HBM 读取，而是在相邻 cells 之间传播、复用多次。TPU 的核心 Data Orchestration，就是把矩阵乘中最频繁的 movement 固化为 spatial wave。

代价也很明显：结构过于规则。Shape 很小、映射不合适或 sequence 很短时，fill 和 drain 占比会很高，array 难以长期保持饱满状态。

## 二十二、Graphcore IPU：显式的 Compute–Sync–Exchange（82:39—85:31）

Graphcore IPU 由许多个 IPU tiles 组成。每个 tile 是一个硬件单元，其中有 processing core 和 local in-processor memory。计算核心可以直接使用旁边 local memory 中的数据；如果下一个 operation 放在另一个 IPU tile，就要通过 exchange fabric 搬过去。

官方给出的执行模型由多个 supersteps 构成。

1. **Compute**：每个 IPU tile 只对自己 local memory 中的数据执行当前阶段 program；
2. **Sync**：相关 IPU tiles 到达阶段边界，确保没有 tile 仍在使用即将交换或覆盖的数据；
3. **Exchange**：数据按预先规划的通信关系经过 fabric，移动到下一阶段所需的 IPU tile。

中间的 sync 与我们前面一直讨论的“buffer 何时可复用”很相似。

这种组织的优势，是 local work 与 communication phase 非常清晰。劣势也对应得很直接：placement 不好就需要不断 exchange；work partition 不均衡时，sync 要等待最慢的 IPU tile；某个 tile 的 local memory 不够，也会限制能放进去的数据和 program。

所以 IPU 的 Data Orchestration，是把数据放在哪里、何时交换数据这两件事做得极其显式。

## 二十三、Groq TSP：Functional Slicing 与静态时序（85:31—90:15）

最后一个是 TSP，这里指 Tensor Streaming Processor。论文用一张图比较两种组织芯片的方法。

传统 multicore 中，每个 core 内含 instruction control、integer/floating-point arithmetic、load/store 和 network interface 等多种功能，不同 cores 再通过 on-chip network 交换数据。也就是说，每个 core 在功能上相似，而一个 core 内部包含多种功能。

TSP 把组织方式反过来，称为 functional slicing：相同类型的 tile 沿一个方向组成一条 slice。不同 slices 分别负责不同功能：

- MEM：memory read/write；
- VXM：vector arithmetic；
- MXM：matrix operations；
- SXM：stream permutation、lane shifting 或 routing；
- ICU：instruction control。

论文把传统 multicore 描述为：每个 core 内部 heterogeneous，但芯片重复 homogeneous cores。TSP 则是每条 slice 内部很 homogeneous，整颗芯片由不同功能 slices 组成。

图中两种流动方向很有意思。Instructions 从 ICU 沿 functional slice 单向穿过；data flow 则在不同 slices 之间横向流动。Instruction flow 和 data flow 使用不同方向。数据不必先被某个完整 core 拿走计算，再通过通用 network 发给下一个完整 core。

把一行硬件拆开看，stream 仍然连接 producer 与 consumer。MEM 从 SRAM 读 operands，并把最终 result 写回 memory；SXM 对 vector lanes 做 transpose/permutation；VXM 做 element-wise vector operations；MXM 做 matrix multiply-accumulate；下面的 ICU 为各 functional slices 提供 instruction streams。

这种架构带来一个困难：它对时机要求很高。TSP 不依赖 GPU 那种动态 eligible-warp scheduling——GPU 可以判断 warp 是否 ready，ready 后再取 operand 发射。TSP 把大量责任交给 compiler。

Compiler 必须安排：数据经过哪些 slices，指令在哪个 functional slice 执行，instruction 与 operand 在哪个 cycle 相遇，以及各种 streams 会不会冲突。这是非常复杂的静态调度问题。

## 二十四、总结：不确定性被放在哪里（90:15—93:00）

总结这四种 accelerator，它们的主要区别可以问一句：**Where does the uncertainty live?** 也就是 orchestration 的责任交给了谁。

GPU pipeline 由 hardware scheduler 动态选择 eligible warps；程序员或编程 framework 显式管理 stages、barriers、waits、不同 warp roles，以及 descriptor 如何构造。优化时重点看 buffering、occupancy 和 eligible warps。

TPU 的 systolic array 把密集 cell reuse 和 operand forwarding 固化为空间 wave。优化时主要看 array mapping，以及 fill/drain 的比例。

Graphcore IPU 把 code/data placement 与 compute–sync–exchange loop 做得极其显式。优化时要做好 placement、降低 exchange volume，并处理 slowest-tile 问题。

TSP 把计算资源组织成 functional slices，让 compiler 安排 instruction 与 operand stream 相遇的时机。优化重点是 functional-slice utilization 与 compiler scheduling。

| 架构 | Orchestration 的主要形式 | 优化时重点关注 |
| --- | --- | --- |
| GPU pipeline | 动态 eligible-warp 调度，软件显式管理 stages 与 waits | Buffering、occupancy、eligible warps |
| TPU systolic array | 空间 wave、规则 MAC cells 与 operand forwarding | Array mapping、fill/drain |
| Graphcore IPU | 显式 placement 与 compute–sync–exchange | Placement、exchange volume、slowest tile |
| Groq TSP | Functional slices 与静态 stream schedule | Slice utilization、compiler scheduling |

这些 trade-off 不一定是缺点，更像是在告诉我们：使用不同 accelerator 时，优化工作应该看向哪里。

四种架构都没有从源头消灭 Data Orchestration 中的 movement、storage、ordering 和 scheduling 问题。它们只是重新划分程序员、compiler、runtime scheduler 与 data path 各自承担的责任。

好，今天就讲到这里。

---

## 附：校订与听辨说明

1. 主讲人姓名依据视频简介采用“卢怡霏”，英文封面为 Yifei Lu。
2. 自动字幕对 CUDA 术语误识别较多，例如把 pipeline、CTA、shared memory、Nsight Compute、TMA、`mbarrier`、proxy、persistent kernel 等识别为近音词；正文已结合上下文和视频画面恢复。
3. 代码是按讲者展示和口述恢复的示意代码，用于表达控制流与依赖关系，不是完整可编译程序。
4. Pointer-chasing 实验中的 RTX 5090、`170 × 32` launch configuration、199.85 warp cycles、193.1 Long Scoreboard cycles、96.6%、0.01 eligible warp、99.5% No Eligible、2.10 ms 与 2.81 ms 均按视频保留。讲者明确说明该实验是刻意构造、并不严谨，主要用于解释依赖链与延迟隐藏。
5. 在 TSP 部分，原字幕于 01:29:38—01:29:45 存在约 7.7 秒空白；正文未凭空补写该处，只保留空白前后能够确认的论述。
6. SM80/SM90/SM100 接口的具体约束会随 CUDA 与 PTX 版本变化，实际编程时应以目标架构对应的官方文档为准。
