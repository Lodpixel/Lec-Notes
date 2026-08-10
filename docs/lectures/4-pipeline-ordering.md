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

本转录以 B 站 ai-zh 自动字幕为底稿，结合视频画面按语义校订整理。字幕覆盖全片（00:00:00.160—01:33:00.900），因此未重新运行 ASR。正文保留完整知识脉络，同时删除口头重复，恢复 `cp.async`、TMA、`mbarrier`、warp specialization、persistent kernel、CLC 等术语以及公式和伪代码。原始字幕与审阅媒体保存在本地 `course-lecture-builder/jobs/BV1NRuX6UEwJ/raw/`，不随网站发布。

---

## 一、Data Orchestration 到底在组织什么

这一讲讨论 Pipeline Ordering。所谓 data orchestration，不只是“把数据搬过来”，而是同时回答四类问题：

1. 数据从哪里移动到哪里；
2. 中途存放在哪一级存储；
3. 谁生产、谁消费，以及双方何时同步；
4. 多份工作以什么次序发射、完成和复用资源。

以 GEMM 为例，一块数据会经历 HBM、shared memory、寄存器或 TMEM、tensor core，最后再由 epilogue 写回 global memory。HBM 访问延迟很长，shared memory 和寄存器容量有限，而 tensor core 消耗数据极快。优化不能只盯着某一条加载指令，而要安排整段旅程。

核心办法有两个：**复用**与**重叠**。复用让搬进来的数据多算几次；重叠则让下一批数据的搬运与当前批次的计算并行。

### Little 定律与在途工作量

Little 定律写作：

$$L=\lambda W$$

其中 $L$ 是系统内平均在途工作量，$\lambda$ 是吞吐率，$W$ 是平均延迟。若一次内存访问延迟很长、又想维持高吞吐，就必须同时保有足够多的在途工作。在 GPU 上，这些工作可能来自：

- 多个 outstanding memory request；
- 多个 pipeline stage；
- 同一线程中的独立指令（ILP）；
- 多个已经 ready 的 warp（TLP）。

所以 pipeline 的本质不是“加几个 buffer”，而是主动制造足够的并发工作，同时确保依赖关系仍然正确。

## 二、从串行 copy-compute 循环开始

最朴素的 tiled GEMM 会重复：把一个 K tile 从 global memory 搬到 shared memory，同步，计算，再同步：

```cuda
for (int k = 0; k < num_tiles; ++k) {
  copy_tile_to_smem(k);
  __syncthreads();       // 数据已写好，消费者才可读取
  mma_from_smem(k);
  __syncthreads();       // 消费完毕，生产者才可覆盖缓冲区
}
```

第一个 barrier 防止 read-after-write：消费者不能在生产者完成写入前读取。第二个 barrier 保护缓冲区的所有权：生产者不能在消费者读完之前覆盖同一位置。即使只有一个 shared-memory buffer，这两个方向的约束也都存在。

异步模型把过程拆成三步：发射工作、执行无关工作、在真正消费之前等待。

```cuda
issue_async_copy();
do_independent_work();
wait_for_copy();
consume_data();
```

`wait` 只证明“被追踪的异步工作已经完成”，并不会自动证明地址正确、参与者一致、跨执行域可见，或旧数据已经安全释放。

### Cooperative Groups：先说清参与者

Cooperative Groups 把参与集合、线程排名和 collective contract 显式化。例如：

```cuda
auto block = cg::this_thread_block();
cg::memcpy_async(block, smem_dst, gmem_src, bytes);
do_independent_work();
cg::wait(block);
consume(smem_dst);
```

还可以用 `tiled_partition` 划出子组，并用 `meta_group_rank` 确定子组编号。任何 collective 都要求约定中的全部线程参与。grid-wide sync 还要求 cooperative launch，并保证相关 CTA 能同时驻留；否则部分 CTA 等待尚未获得执行机会的 CTA，会形成死锁。

## 三、SM80：`cp.async` 与多级流水线

在传统路径中，global-memory load 先把数据放进寄存器，再用 store 写入 shared memory。Ampere SM80 的 `LDGSTS`，即 PTX 层面的 `cp.async`，把这两步合成 global-to-shared 的异步复制，减少显式寄存器中转与指令开销，也更容易与计算重叠。

复制必须满足支持的粒度和对齐。4 字节是基础粒度，常见高效路径按 16 字节组织；地址、范围或对齐不满足契约时，不能仅凭“代码能编译”判断正确和高效。

### Warp entanglement

异步复制的 batch sequence 是 warp 共享的。若 warp 在分歧状态下执行 commit 或 arrive，不同 lane 对序列的认识会纠缠，可能造成过度等待或重复更新 barrier。实务上应让 warp 在收敛状态下完成这些操作。

### `cuda::pipeline` 的所有权循环

一个 N-stage FIFO 不断重复以下状态迁移：

1. `producer_acquire`：取得 empty stage；
2. 向该 stage 发射未来 tile 的复制；
3. `producer_commit`：提交生产；
4. `consumer_wait`：等待最老的 stage 变为 full；
5. 消费者执行 MMA；
6. `consumer_release`：把 stage 归还为 reusable。

双缓冲的结构可写成：

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

完整 pipeline 必须包含 prologue、steady state 和 drain。多一个 stage 能更早预取、给长延迟更多时间，却会消耗更多 shared memory 和控制状态，甚至降低 occupancy。最佳 stage 数不是越大越好。

## 四、为什么“有双缓冲”仍会等待

双缓冲并不保证延迟完全隐藏。复制可能发射得太晚；所有 warp 可能在同一时刻撞到 wait；也可能因为 shared memory 或寄存器占用太高，没有足够的替代工作。

需要区分三个概念：

- **active warp**：已驻留在 SM 上；
- **eligible warp**：当前依赖已经满足，可以发射；
- **issued warp**：这一周期真正获得发射机会。

Occupancy 表示可用于隐藏延迟的容量，不等同于执行单元利用率。很多 active warp 如果都被 scoreboard 卡住，调度器仍然没有 eligible warp。

### Pointer chasing 实验：ILP、MLP 与 TLP

讲座在 RTX 5090（SM 12.0）上用 `170 blocks × 32 threads` 做了受控实验。单依赖链如下：

```cuda
unsigned p = tid & mask;
#pragma unroll 1
for (int i = 0; i < iters; ++i)
  p = next[p];
out[tid] = p;
```

每次加载的地址依赖上一次结果。SASS 中的地址计算和 `LDG.E` 构成长 dependency chain；scoreboard 必须等寄存器 ready，warp 才重新 eligible。分析结果约为每条 issued instruction 199.85 cycle，其中 Long Scoreboard 约 193.1 cycle，No Eligible 接近 99.5%。

若改成两条互不依赖的链：

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

就能同时保有两次内存访问。这里既增加了 ILP，也增加了 memory-level parallelism。单链约 2.10 ms，双链做两倍工作约 2.81 ms，单位工作吞吐接近提升 50%。另一种隐藏方法是让更多 warp 驻留，以 TLP 提供替代工作。

由此可以用四个问题审查任何流水线：

1. **Completion**：异步操作何时算完成？
2. **Visibility**：完成后，消费者所在执行域能否看到数据？
3. **Ordering**：哪些操作必须先于哪些操作？
4. **Ownership**：谁当前拥有 buffer，何时允许复用？

## 五、SM90：TMA 把搬运描述成一个契约

Hopper SM90 的 Tensor Memory Accelerator（TMA）不再让许多线程各自发很多小复制，而由一个 elected issuer 发起 bulk、多维 tensor transfer，硬件完成地址生成。

`CUtensorMapEncodeTiled` 编码的 tensor map 包含：

- 源 tensor 的基址、rank、各维尺寸与 byte stride；
- tile 的 box dimension 与 element stride；
- shared memory 的 interleave 和 swizzle；
- 越界填充值、cache 策略与 L2 fetch granularity（64/128/256 B）。

因此 tensor map 不是普通指针，而是一份传输契约。

### `mbarrier`：线程到达加事务完成

TMA 需要 completion object。`mbarrier` 同时追踪线程 arrival 与 promised transaction bytes；条件都满足后，barrier 才切换 phase，stage 才真正 full。

```cuda
using barrier = cuda::barrier<cuda::thread_scope_block>;
__shared__ barrier bar;

if (threadIdx.x == 0) init(&bar, blockDim.x);
__syncthreads();

if (is_elected())
  cuda::memcpy_async(smem, gmem,
                     cuda::aligned_size_t<16>(bytes), bar);

auto token = bar.arrive();
bar.wait(std::move(token));
consume(smem);
```

同一 barrier 可在循环中反复使用。arrival token 标识高级 API 的 phase；在底层 PTX 中常用 parity bit 区分相邻 phase：

```cpp
auto native = cuda::device::barrier_native_handle(bar);
int parity = 0;
for (...) {
  (void)cuda::ptx::mbarrier_arrive(native);
  while (!cuda::ptx::mbarrier_try_wait_parity(native, parity)) {}
  parity ^= 1;
}
```

### 三条边构成一次 TMA 往返

TMA 的 global→shared→global 往返不能只放一个 barrier：

1. **数据 ready**：用 `mbarrier` 等待 global→shared 完成；
2. **跨 proxy 可见**：generic proxy 的写入要通过 `fence_proxy_async` 对 async proxy 可见，并让所有写入者完成；
3. **源 buffer 可复用**：shared→global 的 bulk async read 完成后，才能覆盖 shared-memory source。

```cuda
cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
__syncthreads();

if (is_elected()) {
  cuda::ptx::cp_async_bulk(/* smem -> gmem */);
  cuda::ptx::cp_async_bulk_commit_group();
  cuda::ptx::cp_async_bulk_wait_group_read(
      cuda::ptx::n32_t<0>());
}
```

源程序次序本身不能替代 cross-proxy fence。

### Swizzle 服务于消费者

TMA swizzle 改变“逻辑 tensor 坐标→shared-memory 物理地址”的映射，用来减少 bank conflict。global memory 的最佳连续布局未必是 tensor core 消费时的最佳 shared-memory 布局，因此布局应从消费者访问模式反推。特定 swizzle 也带有固定的对齐与粒度约束。

## 六、Warp specialization：把 CTA 划成协作代理

Warp specialization 让不同 warp group 承担不同角色：

- producer warp group 发射 TMA，并维护 empty/full stage；
- consumer warp group 等待 full stage，执行 WGMMA、维护 accumulator，并 release stage；
- 必要时另设 epilogue consumer。

这样可以把控制流与数学流水分开，为不同角色分配不同寄存器预算，让生产者按自己的节奏填充未来 stage。改变的是“谁执行”，不变的仍是 acquire→produce→wait→consume→release 的所有权循环。

## 七、SM100：TMEM 与 `tcgen05`

Blackwell SM100 引入 Tensor Memory（TMEM）存放 accumulator，缓解寄存器压力。`tcgen05` 由 elected thread 发射；A 可来自 shared memory 或 TMEM，B 来自 shared memory，结果累加到 TMEM，并可选择 two-CTA cooperation。

新的资源也带来新的生命周期：

- TMEM 何时分配和释放；
- MMA 与 accumulator 访问如何排序；
- `LDTM`/epilogue 如何接管结果；
- two-CTA pair 如何协作并一致终止。

硬件自动化了数据通路，但没有消除 completion、visibility、ordering 与 ownership。

## 八、MoE：不规则工作上的双层流水线

Mixture of Experts 会把 token 路由到不同 expert。每个 expert 收到的 token 数不同，因而形成许多 M 不同、tile 数和执行时长都不同的 GEMM，尾部负载不均衡尤其突出。

常见组合是 grouped GEMM 加 persistent kernel。普通 CTA 完成一个 output tile 就退出；persistent kernel 只启动受控数量的 worker，让每个 worker 反复领取下一个 tile，从而摊薄初始化成本，并复用内部的 TMA↔MMA 流水线。

```cpp
WorkerState state = initialize_once();
while (scheduler.next_tile(work)) {
  bind_problem_and_coordinates(state, work);
  compute_one_output_tile(state, work);
  finish_epilogue_and_release(state, work);
}
drain_and_exit(state);
```

这里存在两层 pipeline：外层跨 tile/problem 调度，内层沿 K 维在多个 stage 间搬运和计算。状态也应分层：

- per-worker：角色、barrier、scheduler；
- per-stage：shared-memory slot 与 phase；
- per-tile：坐标、指针和 epilogue 参数。

Persistence 会长期占有资源，可能影响其他 kernel 共存；终止和 drain 更复杂；静态映射不佳时仍会负载失衡。Ping-pong 方案可让一个 producer 配两个 consumer warp group，交替执行 MMA 与 epilogue，使一个 tile 的 epilogue 和另一个 tile 的 MMA 重叠，但会增加 live state 和退出协议的复杂度。

## 九、CLC 与 grouped GEMM 调度

固定 work-per-block 有利于负载均衡和抢占，但启动开销高；固定 block 数量开销低，却可能难以均衡。Blackwell 的 Cluster Launch Control（CLC）让 resident block 取消并“窃取”尚未开始的逻辑工作，试图兼顾低启动开销、负载均衡与抢占机会。

取消请求本身也是异步生产者。因此应在处理当前 tile 时提前发出请求，以隐藏获取下一份工作的延迟。

对 grouped GEMM，第 $g$ 个问题的 tile 数为：

$$
n_m^{(g)}=\left\lceil\frac{M_g}{B_M}\right\rceil,\qquad
n_n^{(g)}=\left\lceil\frac{N_g}{B_N}\right\rceil
$$

$$n_{tiles}^{(g)}=n_m^{(g)}n_n^{(g)}$$

把各组 tile 数做前缀和，就能把全局 tile id 映射回 problem 与局部坐标。例如三组分别有 6、2、6 个 tile，则全局 id 9 位于第三组的区间 $[8,14)$。

一种简单的静态 persistent 调度是启动 `NUM_SM` 个 program，让 program $p$ 依次处理：

```text
p, p + NUM_SM, p + 2 * NUM_SM, ...
```

它无需全局队列，但“相同 tile 数”不等于“相同时间”：边界 predicate、数据类型转换、epilogue 复杂度和 cache locality 都可能不同。动态 work queue 或 CLC 可以进一步改善尾部。

跨 problem 边界时，pipeline 的角色和 stage 协议可以继续存在，但必须重新绑定 group id、A/B/C/D 指针、stride、descriptor、predicate 和 epilogue destination。不能只更新一个基址就假设其余状态仍然有效。

## 十、其他加速器把不确定性放在哪里

### TPU：脉动阵列

数据以 wave 的形式注入 systolic array，经历 fill、steady state 和 drain；MAC cell 在空间上复用并转发 operand。规律的大矩阵适合这种数据流，但小 shape、差 shape 或短序列会让 fill/drain 占比过大，阵列利用率下降。

### Graphcore IPU：显式放置与 BSP

IPU 由许多 tile 组成，每个 tile 有计算核心与本地内存，程序按 compute→sync→exchange 的 Bulk Synchronous Parallel superstep 前进。放置和交换很显式；代价是放置不佳会造成大量通信，负载不均时所有 tile 等待最慢者，本地内存也构成硬限制。

### Groq TSP：静态 tensor streaming

TSP 按功能切片，而不是复制许多通用核心。切片包括 MEM、向量 VXM、矩阵 MXM、switch/permutation SXM 与 instruction control。指令沿切片单向流动，数据横向 streaming；编译器必须安排使用哪些切片、在哪个 cycle 让指令遇到 operand，并避免 stream conflict。它不像 GPU 那样依赖动态 eligible-warp 调度，因此更多责任转移给静态编译计划。

| 架构 | 不确定性主要在哪里 | 主要代价 |
| --- | --- | --- |
| GPU pipeline | 动态 ready 状态与显式 wait | buffering、occupancy 与协议复杂度 |
| TPU systolic array | 空间 wave 与规则 MAC 阵列 | shape 刚性、fill/drain |
| Graphcore IPU | placement 与 BSP exchange phase | 映射质量、全局负载不均 |
| Groq TSP | 功能切片与静态 stream schedule | 编译器负担、较少运行时动态性 |

没有哪种架构真正消除了 orchestration；它只是把责任在程序员、编译器、运行时调度器和数据通路之间重新分配。

## 十一、总结：用四个问题设计流水线

从 SM80 的 `cp.async`，到 SM90 的 TMA 与 `mbarrier`，再到 SM100 的 TMEM、`tcgen05` 和 CLC，硬件不断降低地址生成、搬运与动态调度的成本。但正确的 pipeline 始终要回答：

1. 操作何时完成；
2. 数据何时对下一执行域可见；
3. 生产、消费和复用必须满足什么顺序；
4. 每个 buffer、stage 与逻辑 tile 当前归谁所有。

性能层面则回到 Little 定律：通过更多 outstanding request、pipeline stage、独立指令或 ready warp，提供足够的在途工作；同时控制 shared memory、寄存器/TMEM、barrier 和 occupancy 的成本。

真正可维护的高性能 kernel，不是堆叠最多的异步原语，而是让每一次状态迁移、每一次等待和每一次资源复用都有明确契约。

---

## 附：校订与数据说明

1. 主讲人姓名依据视频简介采用“卢怡霏”，英文封面为 Yifei Lu。
2. 视频字幕把部分指令名、架构名和分析指标音译，正文已结合画面恢复；示例代码按讲义语义排版，并非可直接编译的完整程序。
3. Pointer chasing 的设备、时延与 profiler 数值按视频画面和讲述保留，主要用于解释依赖链，不应直接外推到其他 GPU。
4. CLC、TMA、TMEM 与 `tcgen05` 的具体接口和约束会随 CUDA 工具链演进，实际编程应以目标架构对应的 CUDA/PTX 文档为准。
