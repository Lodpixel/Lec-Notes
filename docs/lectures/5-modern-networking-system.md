# 5 Towards Modern Networking System —— 完整讲课转录

## 视频信息

- 标题：北京大学未名超算队 × LCPU AI Infra Seminars 系列讲座：5-Towards Modern Networking System
- 链接：https://www.bilibili.com/video/BV1Xg836FE7G
- UP 主：北京大学Linux俱乐部
- 主讲人：王志豪
- 时长：2:45:15
- 讲座日期：2026 年 8 月 17 日
- 活动背景：2026 年暑期 Weiming HPC Training Camp × LCPU AI Infra Seminars 第五次活动录播

## 校订说明

本转录以通义听悟生成的 3610 条中文字幕为底稿，结合视频画面按语义校订。字幕覆盖 `00:00:00.500—02:45:11.440`，与全片时长一致。正文严格沿用原视频的讲述顺序，保留授课内容、推导、例子、现场过渡与必要的重复说明；只删除纯语气词和无意义的口头重复，并恢复 UART、credit、FLIT、PCIe、InfiniBand、Ethernet、TCP、RDMA、UEC、tile 等被误识别的技术词。小标题和时间范围为方便阅读而补充，不改变讲者的论证次序。

为便于逐句检查，另提供[原文逐段对照稿](5-modern-networking-system-source.md)：它保留全部 3610 条字幕及原顺序，只清除了不可见字符，没有替换自动转录中的错字和术语误识别。

---

## 一、开场：怎样从电路走到现代网络（00:00—06:00）

今天给大家介绍网络专题，标题是“现代网络系统结构的展望”。这次会从头介绍整个网络的构建过程。标题页上的日期写错了，今天是 17 号；第一页还留了我的邮箱，大家之后有问题也可以发邮件询问。

整场讲座分三部分。第一部分介绍基础：从电路、信号和链路开始，一直走到程序员手里可以管理的网络资源。第二部分讨论近年网络微架构的发展方向。最后再问一个更远的问题：未来的系统结构可能发生什么变化，网络又该怎样现代化，尤其是 tile-based computing 会给网络带来什么影响。

先用 NVIDIA GPU 做一个有趣的开场。不同人看 GPU，会看到完全不同的东西。看财报的人会觉得数据中心业务像印钞机，收入和毛利率都很高。看成本的人会开玩笑说这是“买内存送芯片”：GPU die 本身的估算成本并不占大头，HBM 和先进封装已经很贵，但成品还能卖到几万美元。现在 NVIDIA 的整机和板卡又常用金色装饰，看起来真像黄金。于是形成一个循环：有钱就买 GPU，有 GPU 就生产 token，有 token 就能卖钱，再用钱买更多 GPU。

但研究者和设计者不能只从价格看 GPU。我们要问的是它由什么组成、各部分怎样组织。芯片层面有许多计算 cluster、cache、片上网络接口、memory controller 和 HBM；系统层面可能是一颗 Grace CPU 连着两颗 GPU，再连 NIC 和存储；更大一层又把很多节点组装成机柜。缩小到 tensor core，一次矩阵乘也要先读数据，把数据送进 shared memory，再进入乘累加单元和 accumulator。

系统结构研究的就是怎样组装这些东西，以及它们之间如何发生关系。这个过程很像盖楼：先有骨架和砖块，再布置水、电等基础设施，最后让各部分一起工作。说得简单一些，就是先在纸上画方块、用线连接、给方块写上名字，最后流片，变成亮晶晶的芯片。面包板和 B200 看起来差别巨大，但背后的系统化思维是一致的。

## 二、从电报、TTY 到 UART（06:00—12:06）

通信的历史很早。从有线电报开始，人们就在一根线上用开关控制电流，把信号从 A 送到 Y。电流必须形成闭合回路，早期系统常把大地当作回流导体，这也是 ground、0 V 参考电位这些概念的历史来源之一。但大地的阻抗会变化，信号并不可靠，所以后来还是使用两根明确的导线。

电报最早除了军事用途，重要客户就是金融业。1869 年已有 stock ticker：一组脉冲驱动机构逐步转动，另一组动作把股票代码和价格打印到纸带上。机械装置如果多走或少走一步就会失去同步，因此当时已经在研究自动同步与恢复。

后来有了 teleprinter/teletype，人们把打字机接到电话线路上。拨通以后，键盘输入通过音频线路发送，对端把收到的字符打印出来。电子计算机出现后，这一形态自然变成 time-sharing terminal：许多人各自使用一台只负责 I/O 的小终端，通过线路共享远处的大型计算机。Linux 到今天仍保留 `TTY` 这个名字，正是这段历史的遗留。

串口 UART 可以看作这套机制的直接延续。线路空闲时保持高电平，这样测到持续低电平时还可以判断线路可能断开。发送方先拉低一个 start bit；接收方看到下降后，按双方约定的 baud rate 定时采样，依次读取 D0 到 D7，最后用高电平的 stop bit 结束。早期拨号上网就是把计算机串口接到 modem，由 modem 在电话线上调制这些数字信号。

## 三、握手、流控、credit 与可靠性（12:06—20:10）

只传 DATA 会遇到两个问题：发送方不知道当前数据是否有效，也不知道接收方是否有能力接收。于是我们给通路增加 `VALID` 和 `READY`：

- `VALID=1` 表示 DATA 上确实有一个有效值；
- `READY=1` 表示接收方现在有空间接收；
- 只有二者同时为 1 时，传输才真正发生；
- 如果 `VALID=1` 而 `READY=0`，生产者必须保持数据不变，不能擅自换成下一个值。

短距离电路里，这种握手很自然；线路一长，READY 返回也需要时间。接收方今天说“别发了”，这句话到发送方之前，链路上已经有一批在途数据，不能凭空停住。因此有两类思路。

第一类是先验许可：接收方预先告诉发送方还有多少 buffer，发送方每发一份就消耗一个 credit，credit 用完便停止，等接收方释放槽位后再归还 credit。这是 lossless flow control。旋转火锅可以帮助理解：每只盘子就是一个 credit。后厨只有拿到空盘才可以继续上菜；顾客取走菜、归还空盘，后厨才得到新的发送资格。路越长、带宽越高，在途的盘子就越多，因此需要覆盖 bandwidth-delay product 的足够缓冲。

第二类是后验恢复：发送方先乐观地发送；接收方来不及处理就丢弃，发送方保留副本，之后根据 ACK、超时等信号重传。这里不能简单地说“可靠网络一定好、不可靠网络一定差”。把状态和 buffer 放在接收端，还是放在发送端，选择的是不同的成本、规模和故障模型。

由此可以把网络按是否丢包、是否可靠地恢复来理解。可靠性不等于无损流控：一条链路可以允许中间丢包，但通过端到端重传最终可靠；也可以逐跳不丢包，却仍然需要处理比特错误和设备故障。

## 四、PCIe：逐跳无损、重放与 Nullify（20:10—31:10）

PCIe 是一个适合观察这些机制的 case。它通常运行在机箱或 PCB 内，拓扑和链路条件相对可控，误码率低，因此 Data Link Layer 使用 credit-based flow control。接收端会分别通告不同类型 TLP 的 header/data buffer credit，例如 Posted、Non-Posted、Completion，以及对应 virtual channel。初始化时通过 `InitFC` 建立额度，运行中再用 `UpdateFC` 归还可用空间。发送方只有持有相应 credit 才能发包，所以不会把下一跳 buffer 冲垮。

无损流控并不能阻止物理误码。PCIe 仍给包加 LCRC，并在链路两端维护 sequence number、ACK/NAK 和 replay buffer。若第 3 个包出错，即使 4、5、6、7 已经到达，也可以要求从 3 开始重放，这就是 Go-Back-N。PCIe 因而形成一种逐跳 lossless、逐跳 reliable 的互连；经过 switch 时，每一跳都执行自己的 credit 和 replay 协议。

这里还有一个有趣的问题。较大的 TLP 经过交换机时，为了降低延迟，交换机可能读取 header 后就开始 cut-through forwarding，不等整个包和末尾 LCRC 全部到齐。假如最后才发现 CRC 错误，前半个包已经发出，不可能把信号从线里收回来。PCIe 的处理是 Nullify：在包尾把该 TLP 标记为无效，让下游知道此前收到的内容必须丢弃。它体现了一个普遍事实：串行链路上“已经发生”的物理传输不能撤销，只能追加状态来否定或补偿。

链路设计也受引脚数量约束。固定 pin budget 下，可以做更多窄端口，也可以做更少宽端口。讲者举了 Google silicon photonics 的例子：面对旧节点 100G、新节点 200G 的混合部署，希望在固定光纤资源下动态重新分配链路，而不是让端口宽度从制造时起就永久焊死。

## 五、从交换器到 router：HOL blocking、VC 与 FLIT（31:10—46:40）

把一组接收 FIFO、发送端口复制四份，中间加仲裁器和 crossbar，就得到一个基本交换器。若采用 input-queued switch，每个输入只有一个 FIFO，会出现 Head-of-Line blocking：队首包想去的输出端口正忙，排在它后面的包虽然想去空闲输出，也无法越过队首。经典分析给出的极限吞吐率约为 58.6%。

解决办法是把一个物理输入拆成多条逻辑队列，例如 Virtual Output Queue 或 Virtual Channel，按目的端口或 traffic class 分开。它们仍共享同一条物理线和 buffer，但阻塞关系被隔离。类似地，可以把低延迟 SSH 和 bulk download 分开，也可以把 request 与 response 分开，避免因协议依赖形成死锁。

典型 input-buffered router 的流水线包括 routing computation、VC allocation、switch allocation 和 crossbar traversal。Intel TeraFLOPS 芯片是一个具体例子：许多小 core 排成二维 mesh，每个 tile 的 router 有 north、east、south、west 和 local 端口，并使用两个 VC。不同 tile 之间还可能跨 clock domain，需要 synchronizer 处理相位差。

buffer 不一定真的为每个 VC 物理切开。可以做一个共享存储池：新 flit 写入 free slot，根据 routing/VC 结果获得一个“颜色”，再通过 linked list 串到对应队列的 head/tail；出队时更新 head 并释放 slot。这样不会出现一个 VC 已满、另一个 VC 的保留空间却闲置的问题。讲者展示了一个五阶段 shared-queue router pipeline 来说明这一点。

网络里还要区分 message、packet 和 FLIT。message 是软件想表达的完整事务，可以被拆成多个 packet；packet 又可被路由器拆成 flow-control digit，也就是 FLIT。FLIT 是交换和流控的基本单位。一个 packet 在 wormhole router 中像一条虫子：head 先决定路径，body 跟随，tail 释放资源。若接口只有 valid/ready，还要用 `LAST` 标记 packet 边界；否则接收端不知道连续数据到哪里才算一个完整包，也不能随意在半个包中插入另一包。

最后看拓扑。indirect network 把 endpoint 和 switch 分开；direct network 则把 router 集成进计算节点，节点之间构成 ring、mesh、torus 等拓扑。拓扑并不是一张抽象图，它决定需要多少端口、每个包走几跳、故障后还能否绕路。

## 六、因果、乱序、死锁与 Orderlock（46:40—56:50）

网络真正能保证的是因果关系，而不是所有事件的绝对时间和全局顺序。先发 request、收到后再产生 response，这两件事存在 happens-before；两组互不依赖的绿色、蓝色消息走不同路径，则完全可能乱序。若一定要强制顺序，最简单的办法是每发一个请求就等响应回来再发下一个，但这会把并行性和带宽都牺牲掉。

因果依赖一旦形成环，就可能 deadlock。比如两端都把同一条 lossless channel 填满 request，又都必须在这条 channel 上发送 response 才能继续，双方就永久等待。包不断绕圈、始终不能到达或提交，则是 livelock。把 request/response 放进不同 VC，正是为了打断这种资源依赖环。

讲者随后提到 SIGCOMM 2025 的 Orderlock：在 bounded buffer 下，下面三件事不能同时全部获得——losslessness、in-order commit，以及对 out-of-order packet 的 retention。系统必须放弃至少一项。

- PCIe 选择 lossless 和有序提交，不保留后到的乱序包；Go-Back-N 会丢掉后续包并重新发送。
- TCP 运行在会丢包的 IP 网络上，向应用交付有序 byte stream，同时可以在接收端暂存乱序数据并选择性确认/重传。
- 现代 InfiniBand 为使用 adaptive routing，需要处理乱序到达；它的具体取舍与传统严格顺序路径不同，不能再把三项性质想当然地同时满足。

这不是“协议写得够不够聪明”的问题，而是有限资源和因果关系带来的约束。

## 七、InfiniBand：从系统总线到网络（56:50—65:30）

InfiniBand Trade Association 在 1990 年代末形成。当时的构想是把系统总线向机箱外、机柜外延伸：多个计算节点共享网卡、存储和其他 I/O，构成一台更大的计算机。Dell、Microsoft、Intel、Sun、IBM、HP、Compaq 等公司参与了早期推动。后来 InfiniBand 在 HPC 领域胜出，一个重要原因是它把关键数据通路做成廉价、低延迟、可预测的 fixed-function hardware，并适配由少量用户统一管理的小型集群。Mellanox 后来成为核心厂商，再被 NVIDIA 收购；协议规范本身是公开的。

InfiniBand 链路使用逐跳 credit 保证无损，transport 再做端到端可靠性。软件提交的是 message，硬件把它分成 packet；VL 用于隔离不同 traffic。一个多包 message 的 opcode 会区分 First、Middle、Last 和 Only。

RDMA Write 的 RETH 中包含 remote virtual address 等信息，但为了减少 header 开销，地址信息可以只在 First packet 中出现。后续 Middle/Last packet 依赖前包及有序语义，接收端才能知道应该写到哪里。如果 First 丢失，后面的包即使到达也无法独立放置。这种设计在稳定、严格管理的网络里很高效；换到多路径、乱序网络，就会限制独立调度。另一条现代化路线是让每个 packet 携带足以独立执行的 metadata，以更多 header 成本换取 packet-level scheduling。

前面从 receiver FIFO 和 credit 出发，构建了“先验、有损失约束”的网络。反过来，也可以从 sender window、replay 和 acknowledgment 出发，构建允许丢包、依赖后验恢复的 switch。这两种世界观后面还会再次出现。

## 八、domain、connection 与系统边界（65:30—75:10）

接下来讨论管理域、scale-up 和 scale-out。现实中的公司会按部门隔离网络和权限：能访问公开网站，不代表能访问财务部门内部资源。这里的 domain 指一组共享状态、共享信任和管理权的设备；它们可以被当成一台更大的机器。domain 之间则不能默认共享，需要显式创建带生命周期、资源和权限的 virtual connection。

domain 可以嵌套，也可以合并成更大的 domain；虚拟机可以把一台物理机器拆成多个管理域，分布式系统也可以把多台机器聚合成一个逻辑域。操作系统本质上是内存与资源的管理权威，多个操作系统也可能协作管理分布式资源。

GB200 NVL72 是一个例子。一个机柜包含 18 个 compute tray、每 tray 两个 node；每个 node 由 CPU 与两颗 GPU 构成，合计 36 个 OS node、72 颗 GPU。NVLink 让机柜域内 GPU 能共享或访问统一的地址空间，整个机柜很像一台 scale-up computer。多个机柜再通过 InfiniBand 等网络 scale out，此时必须显式管理 connection。层次结构来自物理布线密度、距离、延迟和故障边界：一颗 GPU 坏了、一个 node 坏了、整个 rack 坏了，影响范围显然不同。

讲者用“状态和计算”重新描述计算机：memory bits 是 state，computation 是 state transition function。如果模型大到无法全部放进片上 SRAM，有两种“溢出”。时间维度上，把计算拆成指令流，分段加载和执行；空间维度上，把状态和计算分布到更多节点，形成分布式系统。

存储像图书馆的书架。拆成很多 bank 可以并行取书，但同一 bank 的请求会排队，形成 bank conflict；资源分散得越广，并行访问机会越多，布线、面积和同步成本也越高。memory hierarchy 始终在速度、容量、成本之间取舍。

数据移动往往比算术更耗能。讲者举图中量级说明，一次 DRAM access 的能耗可能相当于数千次简单加法；因此现代计算的主要电力并不一定花在乘加器，而花在搬运与保存数据。TPU/MXU 做成 128×128，也是复用收益、shape utilization、面积与供数能力之间的折中。

网络里最昂贵的“运算”常常是 synchronization，因为光速有限。物理距离从 core 内、chip 内、机箱内扩展到 rack 和数据中心，适合的同步机制必须随尺度变化。

## 九、同步的尺度：乱序退休、timer、interrupt 与游戏 tick（75:10—91:40）

在一个 CPU core 内，可以 out-of-order execute、in-order retire，代价相对低。跨 core 时要用 cache coherence、barrier 等，延迟增加到数十或数百纳秒。CPU 与 device 之间通常通过 programmed I/O 或队列提交任务，再轮询或由 interrupt 告知完成，常见尺度到微秒。跨机器做 replication 或 fence，代价继续上升。

对可预测事件，更适合事先安排 timer，再在预定时间 poll；对不可预测事件，interrupt 更自然。火车有时刻表，就不用一直让站务员盯着铁轨；突发事故不知道何时发生，才需要通知机制。这里仍是先验与后验两种思路。

讲者用 PulseAudio 的 timer-based scheduling 做 case。音频按固定 sample rate 消耗数据，本身高度可预测。若每次依赖声卡 interrupt，interrupt latency 和 OS scheduling jitter 会迫使系统准备更大的 buffer，防止 underrun；buffer 大又增加播放延迟。频繁 interrupt 还会产生大量 privilege transition，让 CPU 难以进入 deep sleep，浪费能量。既然下一批样本什么时候需要是已知的，就可以用 timer 主动唤醒并补充 buffer，获得更稳定的低延迟与功耗。

游戏服务器的 64/128 tick 也是固定节拍。客户端会做 prediction，之后再根据服务器结果 reconciliation。若某个“开枪”包错过了所属 tick，到下一个 tick 再重传原动作通常已没有意义。因此游戏网络可能用冗余、FEC 或重复发送，提高本 tick 到达概率，而不是把很久以前的数据可靠重传到未来。

软件 coroutine 也体现相近变化。传统 thread scheduling 依赖 CPU interrupt 和内核调度；若程序员清楚任务结构，可以在自己的 event loop 中安排 coroutine 切换，减少上下文切换，提高能效与软件效率。不能简单地把 interrupt 说成 poll 的“优化版”；二者对应事件是否可预知、状态放在哪里，以及愿意支付哪类成本。

## 十、SQ/CQ：主机与设备怎样交换工作（91:40—95:05）

对于耗时较长的异步 device task，常用 Submission Queue/Completion Queue 模型。队列一般是位于内存中的 ring FIFO。host 先在 SQ 写入多个 descriptor，再向 device address space 做一次 MMIO write，也就是敲 doorbell，告诉设备“有新任务”。device 收到通知后切换状态，通过 DMA 读取 SQ，执行任务，再把 completion 写入 CQ。最后 device interrupt host，或者 host 主动 poll CQ；host 消费 completion 并推进 head。

InfiniBand 中常称 Work Queue。由于发送和接收是两类工作，一个 Queue Pair（QP）包含 Send Queue 和 Receive Queue，并与 Completion Queue 配合；队列里的条目叫 Work Queue Element（WQE）。这仍是两个方向的 producer-consumer ring：host 生产工作给 device，device 生产完成记录给 host。

GPU Direct 一类方案让 GPU 直接参与这条路径，但基本步骤仍类似：准备数据与 descriptor、通知 NIC、NIC 读取并执行、写 CQ，再通知或由 GPU 查询完成。后面讲者会质疑的不是“硬件内部能不能有队列”，而是是否应该把这种软件可见队列固定成所有 xPU 必须共同遵守的编程契约。

## 十一、Ethernet：从 PHY、MAC 到交换机（95:05—111:10）

Ethernet 已经成为通信基础设施。卫星、监控、船舶通信、抗震救灾、数据中心和移动设备都在使用它。早期构想最有力量的一点，是 Ethernet 不绑定某一种物理介质：无线、电话线、同轴电缆、双绞线或光纤都可以承载；上层只需要一条能发送 bit 的 channel。

主机通过 Network Interface Card，也就是 NIC，连接 Ethernet。NIC 给软件提供 send/receive queue，并通过 DMA 在主机处理其他工作时把 packet 写入内存。MAC 层负责 framing、地址过滤、介质访问与 error detection；PHY 负责 modulation、encoding 和具体介质的电/光信号。MAC 与 PHY 之间用 media-independent interface 相连，PHY 再通过 media-dependent interface 接铜缆或光模块。

讲者展示了一张电口板卡：RJ45 有四对、八根线，经过网络变压器/共模滤波器，再进入 PHY；PHY 把介质上的编码转换成 MAC 侧的并行或串行数字接口。换成光口则连接 optical module，MAC 以上无需因此重写。

Ethernet frame 在链路上是一个原子传输实体。MAC-PHY 接口可看到 `TX_EN`/valid、data 和 error，却没有能让下游在 frame 中途暂停的 ready。`TX_EN` 从开始到 frame 结束必须连续；若中间断开，PHY 无法知道前后属于同一 frame。FCS 又位于 frame 尾部，只有数据发完才能判断 CRC。若最后发现错误，只能标记/丢弃整个 frame。这就是 Ethernet switch 通常采用 store-and-forward 的原因；某些受控的片上路径可以 cut through，但必须另有办法处理后发现的错误。

最早 Ethernet 是所有主机挂在同一根共享线缆上，大家都能听到所有 frame，不属于自己的便过滤。现代网络改用 switch。每个 NIC interface 有 MAC address，地址通常由 IEEE 分配给厂商的 OUI 加厂商序号组成。

普通二层 switch 可以零配置自学习。它观察每个入站 frame 的 source MAC，记录“这个 MAC 从哪个 port 来”；下次看到对应 destination MAC，就定向转发到该 port。若 destination 尚未知，则对其余端口 flood；目标设备响应后，switch 又从响应的 source MAC 学到位置。因此所谓“傻瓜交换机”插上就能用，不需要先配置完整 routing table。

但 MAC forwarding table 容量有限，可能只能记录几千或几万项。把几十万甚至更多 endpoint 全挂在一个二层 domain，不仅表放不下，广播范围也不可接受。于是需要进一步分层。

## 十二、IP、ARP 与 TCP：从子网走向全世界（111:10—115:05）

IP 给主机分配层次化地址，用共同 prefix 把一组二层设备组织为 subnet。比如 `192.168.0.0/24` 中，前 24 bit 是 prefix，后 8 bit 区分该 subnet 内的 host。掩码与地址按位与得到 prefix。层次命名压缩了查找状态：router 不必为全世界每台机器保存一条平坦 MAC 记录，而可以按 prefix 聚合 route。

不同 subnet 之间通过 gateway/router 跳转。routing table 做 longest-prefix match：匹配到直连 subnet 就从对应 interface 发送；匹配不到更具体 route 时，走 `0.0.0.0/0` default gateway。命名方式决定表的容量，容量又决定管理边界。L1 是 physical layer，L2 是 data link/Ethernet，L3 是 IP。

主机知道目标 IP，却还需要知道下一跳在当前 L2 domain 的 MAC address。ARP 会广播“谁拥有这个 IP”；对应主机单播返回自己的 MAC。结果放进 ARP cache，之后可以直接封装 Ethernet frame。ARP 不是 IP 本身，更像粘合 L2 和 L3 的协议。

IP 只做尽力而为的 packet delivery。中间 gateway busy 时可以丢包；多条路径还可能造成 duplicate 和 reorder。若应用需要可靠、有序的传输，要在端点软件实现，TCP 就承担这项工作。TCP 建立 connection 时做 three-way handshake，使双方确认彼此收到了建连意图和初始状态；关闭时双方分别用 FIN/ACK 结束各自方向，常见表现是 four-way close，也可能合并报文。异常时可用 RST 表示 connection 状态已无效。真实 TCP state machine 比图示复杂得多，这里只做概览。

## 十三、研究的世界图景：InfiniBand 与 TCP（115:05—127:40）

第一部分回顾了今天网络体系从第一性原理怎样逐步搭起来。进入前沿之前，讲者插入一段关于研究动机的讨论：AI Infra 还是很新的领域，表面上许多工作像工程实现，实际上也可能是在构建计算世界的新“宪法”。每个设计决定都有后果，就像登山前要看天气窗口、规划撤退路线。低效程序只是浪费算力，有缺陷的系统会产生修复成本，产品或公司失败则付出更大代价；承担这种选择也正是人的精神价值之一。

讲者引用了爱因斯坦在普朗克生日会上的演讲。科学殿堂里的人动机不同：有人追求智力快感和雄心的满足；还有人是为了逃离日常生活的粗俗、沉闷和欲望束缚，进入客观知觉和思想的世界。更积极的动机，是人总想按照适合自己的方式，画出简化而可理解的世界图景，再试图用这套体系取代并征服经验世界。这样的图景牺牲完整性，换取纯粹、清晰和确定。讲者也提到马赫与普朗克的争论，建议有兴趣的同学继续了解。

做计算系统同样如此。设计者先形成一套关于现实约束的抽象和直觉，再从基本原则构建体系。体系不等于经验世界，却希望比零散经验更有解释力甚至预言力。InfiniBand 与 TCP/iWARP 正好代表两套各自自洽的世界图景。

在 InfiniBand 的图景里，网络是系统总线、片上网络向外的延伸。计算节点、GPU 和共享 I/O 组成更大的单一计算机，通常位于一个受控 domain。QP、credit 和 ordering state 可以常驻硬件；网络无损、故障少，用户数量有限，作业通过 scheduler 分时复用。在这些假设内，IB 的 fixed-function design 很优美、效率也非常高。但它没有把地球另一端的陌生人访问本机 GPU 当作首要场景，扩展到大量 tenant 和超大范围会越来越困难。

TCP/IP 的图景相反：Internet 是陌生人的网络，中间路径是不可信的 black box，只承诺尽力传送 byte；可靠性、顺序和安全都由 endpoint software 补上。正因为不信任中间网络，物理介质和管理者可以极其多样，从而获得把全世界连接起来的扩展性。代价是 TCP 只提供无 message boundary 的 byte stream，software state 依赖大容量 DRAM；到了高速 AI network，逐 byte/flow 的复杂处理越来越难被硬件高效承载。iWARP 是基于 TCP 的 RDMA，同样继承了这套世界观。

问题并不在于哪一派当初选错了，而是今天的世界图景变了。

## 十四、现代网络为什么需要重新分层（127:40—225:20）

首先，endpoint 变了。过去是 CPU 处理 HTTP、下载等数据；现在 accelerator 要直接进入 data plane。CPU 更适合做 setup：建立 connection、配置 mapping、检查 permission；GPU/NPU 等 xPU 直接 issue 数据传输。协议若包含硬件难以实现的动态数据结构，就必须重新设计。

第二，workload 变了。互联网应用常是较细粒度 flow；AI 训练则搬运大块 tensor，频繁发生 aggregation、scatter、all-reduce 和 point-to-point replica copy。计算节点、存储节点和传统 service 希望共用更少种类的网络，降低同时购买 IB switch 与 Ethernet switch 的成本。

第三，scale 变了。大型数据中心可有十万级 node、百万级 tenant，网络里有海量 in-flight transaction；tenant 和 traffic 混杂，reorder 是常态。Google B4 把分布全球的数据中心连接起来，海底光缆可能故障，流量必须 reroute。即使单机也希望插四张 100G NIC 得到接近 400G，需要 multipath aggregation 与 load balance。于是丢包、拥塞、reorder、migration、路径变化不再是罕见异常，而是正常运行条件。

网络 traffic 可以按粒度和耐久性来观察。游戏、直播数据时效性强，错过时间窗口就失效；普通网络消息粒度小但要求可靠；AI transfer 可能一次几十 MiB，却由约 1 KiB 或更大的 packet 组成，需要 per-packet selective retransmission。不存在适用于所有 traffic 的最优网络。

想获得某项性质总要“结账”：

- 用 redundancy 换可靠，就支付额外 bandwidth；
- 保存 connection/reorder state，就支付 SRAM、DRAM 和 scheduling logic；
- 等待重传与确认，就支付 latency；
- 什么都不付，就只能 best effort。

NIC 本身就是专门处理 packet、retransmission、congestion、connection state 的小型计算资源。可靠传输从来不是免费附赠。

domain 半径决定适合的机制。NVLink 只覆盖一个 NVL72 rack，36 个 CPU、72 个 GPU 可以被看成一台机器，域内用低延迟 credit-based lossless path，并共享内存状态。跨 rack、跨 data center 后，默认共享状态和全局同步就不合理；必须显式创建有 lifecycle 和 permission 的 connection，允许一个故障节点被绕过。几千公里范围内做全局同步，物理上就没有意义。

传统 IB/RDMA 的问题是把 connection 的多件事情“焊”在一起：identity、QPN、path、ordering、reliability 和 hardware state 彼此耦合。一个管理域内由中心分配 QPN，单一路径和顺序传输很高效；要支持 multipath 时，却可能需要多个 QP，或要求 switch 对单一 flow 做 packet spraying。connection 数增加又要求更多 SRAM。identity 如果绑死在物理 path 上，reroute 和 migration 都很困难。讲者因此强调：connection 应是 virtual/software concept，不能等同于一条固定硬件通路。

UEC（Ultra Ethernet Consortium）等现代方案试图解除其中一部分约束。它也体现“非 NVIDIA 世界”各厂商的合作：Broadcom 提供 Ethernet switch，AMD 等提供 CPU/GPU，大家把各自能力组合成与 NVIDIA vertically integrated stack 竞争的系统。但仅靠在旧协议上不断叠加功能，仍可能掉进 coupling 的坑：改得越多，state 越多，扩展越困难。

讲者给出的方向是明确分层：

1. **Semantic layer**：向程序表达 IB-compatible semantic，或者更原始、可自定义的 operation；
2. **Connection layer**：管理任务/连接的 lifecycle、resource、configuration 和 permission；
3. **Execution/transaction layer**：把 2 GiB DMA 之类大 operation 分解成 bounded transaction 和 address stream，负责 admission、mapping、issue、completion 与 retirement；
4. **Tunnel layer**：保证单条 tunnel 的 reliable delivery 和 congestion control；
5. **Path layer**：在物理 Ethernet 上完成 packet forwarding 与 routing。

transaction layer 可以把一个 32 KiB transaction 拆成 32 个 1 KiB packet，在 32 条 tunnel 上 spraying；某条 path 拥塞或故障时自动换路。这样 application identity、transaction、tunnel 和 physical path 不再捆死，每层只解决自己尺度的问题。

## 十五、数据中心网络：spine-leaf、东西与南北流量（225:20—231:24）

一个简单网络可以有三个 node、两层 switch。下层是 leaf，上层是 spine。同一 endpoint 到目标可能有两条等价路径；five-tuple 标识 flow，ECMP 可把不同 flow hash 到不同 path。紫色路径断了就走绿色，两条都健康时同时利用，既做 fault tolerance，也做 bandwidth aggregation。

现代 data center 常把计算网络和存储/业务网络区分开。compute node 之间的 collective 和 replica copy 是 east-west traffic；访问集群服务、模型接口和集中存储则是 north-south traffic。

spine-leaf 的层次来自物理局部性和端口规模。若 16 台机器全部直接互连，连接数迅速膨胀；把它们分成四组，每组四台先接 leaf，再由 leaf 接 spine，就用分层近似完成“开方”，控制每级端口数量。机器数量继续增长时，还会出现 core、aggregation、access 等层级。

对外 service 往往在许多服务器上部署。aggregation layer 对入口 traffic 做 ECMP hash，把一千万用户的请求分散到不同机器；这不是 AI collective 里的 aggregation，而是汇聚、分发和 load balancing。存储网络则像把每台机器的 SSD 抽走，集中放到 storage node：host 看到的仍像本地 block device，实际 command 和 data 经 NIC 到远端执行。

传统网络里 north-south traffic 曾更复杂；随着分布式训练规模增长，east-west traffic 也越来越重，网络架构必须同时处理两者。

## 十六、有状态计算与“The Future Belongs to Tile”（231:24—244:07）

最后讨论未来计算对网络的影响。第一个判断是：stateful operation 不可避免。表面上 $Y=A\times B$ 是无状态函数，输入 A、B 得到 Y；但如果每次都从远处读 operands、算完再写回，communication energy 极高。专用算子为了 data reuse，会把 accumulator 放在内部，逐步变成有状态单元。Tensor Core/TMEM 中的 accumulator 就属于算子硬件状态。

一旦多个用户或多个程序共享 stateful unit，切换便不再只是换 instruction。系统要保存 A 用户尚未完成的 accumulator state，恢复 B 用户的 state，像 time slice 一样调度。这不是多加一点自动化就能消失的问题，需要新的计算系统：operator 除了 input/output，还应有 state save/restore path，processor 必须管理这些更复杂的 contract。

矩阵算子的另一部分是地址生成和 layout。硬件按某种固定方式读取矩阵；软件用 shape/layout 描述地址流，就能让同一物理数据以不同排列送进 operator，再按另一 layout 写回。它不是单个 scalar address，而是描述一整块 tile 怎样被生成、变换和消费。

讲者因此认为未来计算会以 tile 为主要粒度。scalar 一次处理一个数，SIMD/vector 一次处理几个数；为了更高能效，CPU matrix extension、GPU Tensor Core 等会一次读入并复用一整块数组。Intel AMX、ARM SME 等都反映这一趋势，软件侧 Triton、TileLang/CuTe 一类 tile-based programming model 也正在兴起。

既然计算和数据都是 tile，网络 traffic 的主单位也会成为 tile/transaction。tile 不应被迫先写入 CPU/GPU DRAM，再由 NIC 从 DRAM 读出。现代数据中心网络已达到几百 Gbit/s、几百纳秒量级，越来越接近 memory path；若所有 data 都经 SQ/WQ descriptor 和 DRAM buffer 中转，DRAM 反而成为瓶颈。

讲者给出一句概括：**CPU owns setup，xPU owns issue**。CPU 配好 connection、address mapping 和 permission；真正的数据面由 accelerator hardware 直接发起。控制与同步只需少量 scalar metadata，大块 tile 在 operator、DMA、collective engine 和 fabric 之间直接流动。

这里并不是说 hardware 内部不能有 queue，而是不应强迫程序员把“管理软件可见队列”当作所有 xPU 的统一契约。让写 attention operator 的程序员同时维护 NIC/NPU queue、WQE、doorbell 和 CQ，会产生巨大的认知与实现成本；做一个专用 all-reduce/collective engine 时再复刻整套 queue protocol，也会浪费硬件。

NVMe 与 persistent memory/Optane 提供过相近启示：storage 越来越快，传统 OS 和 queue stack 的管理开销越突出。Optane 最终受价格、专用性和编程生态等因素影响，但直接在 address space 中访问快速持久介质的思路仍很先进。GPU programming 也已经证明，大延迟、大粒度 operation 可以使用异步 issue 加 barrier/wait 的方式表达；程序只写几行 operation 和 dependency，而不是数千行队列管理代码。

最终的通用系统图景是：NIC 是 scale-out、通向 domain 外部的门户；NVLink/fabric 负责 domain 内互连；系统中还可放 DMA engine、aggregation/scatter/collective engine。软件表达“做什么计算和传输”，硬件负责把 bounded tile transaction 安排到适当资源上。

## 十七、结束与现场交流（244:07—245:15）

讲者总结，今天的内容到这里。主持人感谢这次分享，并询问现场是否有问题。有人问联系方式，主持人说明第一页 PDF 已经提供，课件稍后也会发到群里。由于时间已晚、现场没有更多问题，本次讲座结束，主持人再次感谢主讲人与参与者。

## 听辨存疑与说明

- 原始字幕含大量自动转录误识别及不可见字符，发布正文已清理；技术术语根据画面和上下文恢复。
- 讲者在个别 slide 中引用了论文、书籍和产品图，但视频 contact sheet 无法可靠读出全部小号参考文献，正文没有臆造具体书名或论文作者。
- `Orderlock`、UEC、Intel/ARM matrix extension 等处以讲者现场论述为准；本文是讲课转录，不把其中的观点改写成独立事实断言。
- 结尾主持人对主讲人的口头称呼在自动字幕中识别不清，视频简介明确写明本次分享者为王志豪，因此正文统一使用简介信息。
