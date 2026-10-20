# Mellum 2：模型训练与代码智能

> JetBrains 技术交流讲座的可读转录。标题按录音内容拟定，并非活动官方标题。

## 来源与阅读说明

- **录音来源**：同一场讲座的上下半段，本地文件 `jetbrains-1.m4a`、`jetbrains-2.m4a`；总时长 **50 分 43.627 秒**。原始录音不随本站发布。
- **语言**：中文开场、英文主讲与问答；正文保留原语言，中文小标题由本站整理添加，不是全文翻译。
- **时间轴**：正文使用两段拼接后的累计时间。第二段从 **31:22.603** 开始；回查第二个原文件时，将正文时间减去 **31 分 22.603 秒**。时间戳定位的是整理段落的起点，并非逐字对齐。
- **处理方式**：在本地使用 Whisper **large-v3** 识别，中英文分段；对部分高风险技术表述二次识别，对问答录音进行本地音量均衡后复识别，再交由 DeepSeek 作文字校订。原音频未发送给云端识别服务；DeepSeek 仅接收识别文字及相邻文字上下文。
- **校订边界**：清理语气词、口吃和无信息量重复，保留有效内容、讲授顺序与问答。不据缺失的投影片补写代码、实验图或操作细节，不把讲者的判断改成已证实事实。开场无法核实的中文人名以角色称呼，不猜写姓名。

!!! warning "阅读边界"
    这是自动识别、模型校订并经过文字对照检查的可读稿，不是经讲者确认的逐字稿，也不代表人工逐句听校。现场收音和英文口音使少量专名、短语与数字仍有不确定性；未能可靠恢复的细节不作推断，详见文末编辑说明。涉及实验指标或技术实现时，请对照原始材料。

## 内容导读

以下是本站添加的阅读导航，不替代讲座正文：

- 从代码补全、下一处编辑建议，到具备工具调用能力的编程智能体。
- 预训练阶段的架构、数据与长上下文扩展。
- 指令微调、强化学习、训练与推理基础设施。
- 现场问答：工具调用、推理数据、团队如何使用 AI，以及代码补全与智能体训练的区别。


## 讲座正文

### 开场与校企合作

**00:01**

各位老师、同学，大家下午好！欢迎来到这次 JetBrains 技术交流会现场。我是学生创新中心软件社群相关工作的负责人。我们学校与 JetBrains 公司合作已经长达八年之久。JetBrains 作为知名的软件开发厂商，服务全球上千万的开发者，当然也包括交大的师生同学们。

**00:50**

所以，每年我们都会和 JetBrains 合作推出各类培训跟讲座，来帮助大家更好地使用这些工具。

### AI 与真实开发环境

**01:02**

今年非常荣幸。随着 AI 开始发展，JetBrains 也在实际探索如何把大模型深度融入真实的软件开发环境当中，让 AI 真正服务于我们日常的工程，把 AI 落地到我们的工作环境当中。

### 今日主题：Mellum 2

**01:28**

今天的交流聚焦在 JetBrains 最新开源的重磅成果 Mellum 2。我们非常荣幸地邀请到 Mellum 2 项目的两位核心研发专家 Marko 和 Aral。

**01:48**

接下来，两位嘉宾会为我们完整拆解 Mellum 2 模型，同时分享他们对下一代模型的思路。

**02:01**

下面，我们先请 JetBrains 高校合作与发展部的同事来向我们介绍两位嘉宾以及其他的 JetBrains 同事们。

### 讲座期待：互动与反馈

**02:11**

事实上我先不说太多。很荣幸能够和大家今天交流。但是我们今天的讲座，我希望大家能够轻松一点，因为我们请来的是我们 JetBrains AI 团队的四位核心同事。今天也是希望和大家做一个充分的交流，并不是说一定要分享或者推广我们的工具，更多是想听一下大家的反馈，我们可以做一些更多的互动。这可能是我们今天来这次做讲座的一个期待。

**02:54**

Just a simple example: we don't want this to be a sit-down lecture. We hope for more questions. We hope we have more interaction. And we also hope to hear your views about Mellum AI models. It's really quite an honor to talk with you.

### Aral 开场

**03:13**

Okay, hello everyone, I'm Aral, I'm from JetBrains. Can I get a quick raise of hands: who's heard of JetBrains before? Okay. Who's heard of PyCharm, or IntelliJ, or any of those other code editors? Okay, who's heard of Mellum? Okay, one person. That's why we are here today.

### 为什么要训练代码模型

**03:44**

So, JetBrains makes code editors, like VS Code or other ones. No one writes code by hand anymore, so lately we've been investing a lot into training our own coding models to be used in the editor, but also agentically. And today we are presenting Mellum 2, our latest code model that we trained, which is totally open source. And yeah, so today I'm joined by some of our core training team.

### 训练团队与开放交流

**04:17**

So we have Ivan, who leads our post-training. We have Marko, who leads our pre-training. And Bulat, who leads our reinforcement learning. We are a relatively small team, but we are super open about everything that we do. We are completely open source. We can share any of our experiments. So please, like the host mentioned, do not hesitate to interrupt me at any point, ask questions—the whole point of this is to be interactive.

**04:46**

We want to share the knowledge around large language models, so that this information can be shared publicly. Anyway, that being said, I also have the slides here if you want to follow along.

### 现场提问：手写代码与 agentic 编程

**05:09**

So in the meanwhile, let me ask you a question. Who here writes code by hand anymore or has to do any kind of coding at all? Okay, who writes code by hand? Okay, some people. Who uses agentic models to write code? So just coding using agents. Yeah, okay. Yeah, this is 2026. Using agents is the go-to way for most people to write code nowadays. I think something like 90% of code is written by agents nowadays, so I'm willing to share more about that technology and how to actually train these models and how to use these features.

### 从 Mellum 1 到 Mellum 2

**06:00**

That being said, okay, let's move on. So, quick reintroduction of our names. Also, after the talk, please feel free to come and chat with us—our WeChat, our Twitter; we're super open. Then, let me start out. So last year I was actually also here to present Mellum 1.

Back when we did Mellum 1, we were very focused on creating a model that essentially generates a single line of code. And this is what we set out to do with Mellum 2 as well. Except instead of just generating code, we also wanted it to be able to edit code in your editor. So we have this feature called next-edit suggestion, where, essentially, when you are writing code, it might delete, change, rename something for you. This used to be relevant in 2025, but again, almost no one writes code by hand anymore, so even though we set out to train a next-edit suggestion model, we ended up trying to make it agentic in the end.

**07:08**

By the way, do you think it's possible to turn off these lights?

### 大模型训练流程

**07:15**

Could I also get a quick raise of hands maybe? Who is familiar with the general process of training large language models? Or has seen these stages before? Okay, okay, quite a few. I'm hoping most of you are interested in this because that's what we'll be talking about today.

**07:44**

So, a large language model usually follows two main stages: pre-training, where you feed it a lot of data, basically the entire internet, and you ask it to predict the internet, followed by post-training, which consists of fine-tuning and reinforcement learning.

### 开场与内容概览

**08:03**

So I'll be walking you through everything that we did to train this model—the whole stack—and please feel free to interrupt and ask us questions.

### 什么是大语言模型

**08:15**

So what is a large language model in the end? A large language model—the base model—is trained on the internet, and it essentially tries to predict the next word. You train this with a relatively short context window, which basically means it only sees maybe 8,000 words at a time. After that, you try to predict longer sequences. After that comes the fine-tuning stage, where you try to get it to follow instructions rather than just predict the next word, after which you try to get it to do more complicated things. Sometimes the instruction itself is not fully clear, and you want the model to be smart enough to figure out the things that the person didn't say, for which you're using reinforcement learning.

### 指令模型与思考模型

**09:07**

In these cases we have two kinds of models—I should clarify this: instruction-tuned and thinking models. Instruction-tuned essentially means we want the model to answer immediately. And with thinking models, we give it some time to think, work its way through a problem. This tends to be better for things like mathematics, where you need to do intermediate calculations.

### 预训练基线

**09:36**

Let me get into the pre-training stage. So, when we set out, the best next-edit suggestion model on the market used to be Qwen 2.5. I'm guessing most of you are familiar with or have heard of this company before—Alibaba. They also train open-source language models in a similar size to what we do.

### 目标：更快、更好的补全

**10:00**

And most other code editors already have some kind of code completion or next-edit suggestion, and almost all of them use Qwen 2.5 7B—the 7-billion-parameter model—in some way. So they would usually fine-tune it or do reinforcement learning on this open-source base. So that's what we set out to do, essentially. We took Qwen 2.5 and said, okay, we want this model except faster and better for next-edit suggestion. What did we do here? We looked at how fast you can generate completions with Qwen 2.5, and based on that, we constrained our search space and which parts of the large language model architecture we could change in order to get either faster inference or better quality—ideally both.

### 架构探索

**10:53**

And I will be walking through some of the architectures that we considered. If some of these are not clear, please do not hesitate to interrupt me. We are very deep in this rabbit hole of architectural innovations, I think. But I think it's very interesting to share, and yeah.

### 前馈网络与 MoE

**11:17**

As a first thing, large language models are networks: essentially, you try to multiply matrices together at the end of the day. In this case, we explored whether it makes sense to have one big matrix for our feedforward network or multiple smaller networks that work together. In the end we ended up settling on 64 smaller expert feedforward networks, which are controlled by a router. This router only activates eight on any given token, and based on that you can essentially approximate a much larger network while only activating 2.5 billion parameters per token.

### 分组查询注意力

**12:22**

Next up, we looked into attention networks and how we can optimize those. There's a pretty standard technique from a couple of years ago already, but the interesting thing here is we actually looked at what Qwen was doing for how they applied grouped-query attention, and we essentially found a better trade-off between quality and speed here. Instead of taking Qwen's eight heads, we settled on four attention heads.

### 滑动窗口注意力

**13:03**

Next up—so we continued looking into attention—we looked at another speed improvement, which is called sliding-window attention. So conventional attention is here in pink. When you're predicting the next token, you look at all the previous tokens in the sequence. But with sliding-window attention, you only look at a fixed window of the past tokens, and you can essentially alternate these layers in your network. We stuck with what Google's models did—Gemma models—which is three sliding-window attention layers for every full attention layer.

### 多 token 预测与训练稳定性

**13:44**

Next up, we adopted a cool training technique that was introduced by DeepSeek, I believe, called multi-token prediction. So instead of simply predicting the next token on a large net, on your corpus, on your training data, you can actually predict not only the next one but the one after that at the same time. And DeepSeek actually found this is faster to train, or it yields better performance on your benchmarks for the same amount of training time. And this has a very cool feature: it's not only beneficial for training, but you can also use the secondary token during generation. You can essentially predict two tokens at a time and then on the next pass verify that that second token is correct or not. In our case, it's correct 90% of the time, and this saves us quite a lot of time.

Lastly, we did QK norm. So, in your attention network, you have three matrices—query, key, and value matrices—that get multiplied together, and essentially this is just a training-stability thing. So when you are training, you want to make sure that your attention scores—so how much any token looks at another token—are relatively balanced. Without QK norm, you have the red line; these scores can get very spiky, and you try to get training stability by essentially normalizing the scores before you multiply them.

### 速度结果

**15:48**

So in the end, what did this give us? We set out to beat Qwen 2.5 7B in terms of raw speed. When there's a single user using the model, we basically have a tie. With multiple users making concurrent requests, we are around 30% faster. This is just speed; quality could have become worse, but as I will show later, I think we did quite well on quality too. Before those results, I want to talk more about the pre-training data.

### 预训练数据与优化器

**16:30**

Everything we do in the lab is fully open source. We basically used a lot of open source datasets to train the model, following roughly this curriculum: first train on large-scale internet data, then narrow down to higher quality datasets. What are higher quality datasets? For example, GitHub repositories with many stars, academic papers, etc. In addition, we adopted a new optimizer recently used for training these transformer models, called Muon, also proposed by a Chinese lab, because we found it worked well in our ablation experiments.

### 长上下文与负面结果

**17:26**

After a large amount of web training, you get a base model. You then want it to handle a longer context than it was pre-trained on—ideally, more than the 8,000 tokens we used in pre-training. For this we used a technique called YaRN, which essentially changes the model's positional embeddings to generalize to longer sequences.

We found an interesting negative result here. The speaker refers to Allen AI's OLMo and to adding certain long-text data and synthetic tasks at this stage. We found that this hurt performance and made the model worse, so we did not use that approach.[^olmo]

### 微调

**18:23**

So now we have the base model. Give it some internet text, and it's essentially predicting the next token. How to make it actually follow instructions? Through fine-tuning. In fine-tuning, you give the model a batch of carefully curated demonstrations. What are carefully curated demonstrations? For example, you ask the model a question, and it generates an answer. In the past people actually wrote these datasets by hand; nowadays you can use larger models to generate them. So you come up with many questions, have a larger model generate answers, and then show all this data to the model. This is different from reinforcement learning: in reinforcement learning, the model essentially explores the search space on its own, without a larger model guiding it, and doesn't get much information. So for fine-tuning, we again used a batch of open source datasets. I think we detailed some of them in the paper. But essentially, it's a lot of stuff you can find on Hugging Face. We used different amounts of tokens for each. I mean, a lot of these thinking tokens are essentially just the model trying to plan... obviously, more tokens.

### 微调中有趣的问题

**19:50**

We encountered something interesting during the fine-tuning stage: we hadn't used any Google datasets, and after tuning the model to answer questions, basically whenever you ask it 'which model are you', it always introduces itself as a Google system, even though we didn't use any Google data. So we had to fix this issue. Yes, essentially by adding a bunch of identity Q&A to force the correction... OK, anyway, that was fine-tuning.

### 强化学习

**20:29**

Fine-tuning is basically warming up the model for the most important stage, which is reinforcement learning. So what did we do for reinforcement learning?

### 强化学习基础设施与数据

**20:45**

Well, the problem with reinforcement learning in 2026 is that I think there's no unified solution yet. So we had to build one ourselves internally. We're very proud of these numbers. What do these numbers mean? Completing one reinforcement learning stage takes about two days on our compute cluster. What does that mean? It means we have 10,000 agents working in parallel, trying to solve some task or set of tasks we give it. One run produces millions of trials, it tries to solve various coding problems, and typically we run about 700 steps per training run, during which the model learns and updates its weights.

### 异步性与推理-训练不匹配

**21:41**

So what does that infrastructure actually look like? You have two key components. One is the inference engine that actually runs the model, and then the trainer that updates the model weights. These two are always mismatched with each other. Solving this is a whole other challenge. Why does this mismatch occur? Because when you give a coding agent a problem, two instances of the same coding agent might run for different durations. So you don't want to wait until all thousand agents you're running have finished, because then you'd basically be stuck waiting for the slowest agent. So you need to find a way to get out of this asynchrony. You need to ensure you can periodically execute these steps and update the model weights.

### 并行运行 agent

**22:47**

So how do we do it? First, how do we run thousands of agents in parallel? Obviously this is very compute-intensive. You want to remove any overhead and such costs. We used an open source solution from JetBrains called IDE Gym, which handles these on a compute cluster set up by the research department. Then on top of that, we basically built a validation gateway that hosts all the tests, ensuring the agent actually solved the problem it claims to be solving. So how exactly does this work?

### GRPO

**23:36**

We again borrowed a technique from a Chinese lab, called GRPO. This was proposed by DeepSeek in 2024 at the DeepSeek moment. It essentially runs 16 agents in parallel. It observes how far each agent has progressed in solving the task and rewards the best-performing agent. That agent's weights get reinforced more, and that's the reinforcement learning process. On top of this, we made some modifications to GRPO, published by other labs.

### 奖励塑形

**24:18**

I don't want to get too into the details, but in simple terms, we had to do some reward shaping. Basically, for our instruct models, we want them to answer immediately, so we had to give them lower rewards when they would think for excessively long. Similarly, for our thinking models, we wanted to penalize them if they also spent a very long time thinking, or they didn't use any of the tools that we provided for them.

### 训练与推理不匹配

**24:54**

How did we solve our training inference mismatch? Like I mentioned, there's a problem to solve. Why does it even happen? GPUs that you run these models on are not fully deterministic to begin with. Secondly, two different versions of your inference engine might change the batch size and essentially invoke different kinds of GPU code that results in differences. On top of that, we have a mixture of experts model. So different experts may be activated between the training and the inference engine.

### 解决方案与损失函数

**25:31**

And to fix all of these issues, we basically took the simplest approach possible, which was, when there is a mismatch, just ignore that. There are some other more elaborate techniques that were proposed, but we found this works fine, so that's what we used. We also used a recent loss function from a MiniMax paper.[^loss]

### 模型成果

**26:10**

Let's get to the results. So, what did we get out of all of this? In the six months that it took to train our model and to end with all of the infrastructure and the engineering, we ended up shipping a model that performs, I would say, quite well compared to the best American open source lab that has a model at this size — we, yeah, kind of completely beat it. Against the best Chinese model at this size, we are close to on par. For the thinking model, this is the agentic model with all of our reinforcement learning stuff, we get a model that is, to say shortly, comparable to another 9-billion-parameter model, but at twice the speed.[^comparison].

### 小型工程智能体的前景

**27:19**

What's next? So that's why I'm here. We really believe that small software engineering agents are the future for programming. The cheaper that these agents become — they become ten times cheaper per year — the more that people will use them, and the more accessible they make programming. So we are investing a lot in this, and we believe that open source is the way to go. And for that reason, yeah, we are here, all of us, we are happy to answer your questions, talk to you about the experiments we run or other things, so please do not hesitate.

### 问答：在 Claude Code 中使用 Mellum

**27:55**

Q: I have a question. So suppose I'm running Claude Code and I want to connect with the Mellum. So I wonder how Mellum could work in the Claude Code? I wonder how to operate Mellum on Claude Code and apply it onto Claude Code operations?

A: Yes, so actually we are working on a course that shows exactly how to do this. I think it will come out later this month. The tricky thing about Claude Code is they don't make it easy to switch models. It is possible to do this.[^integration]

### 规划与实现的分工

**28:35**

So the way — the industry accepted way at the moment — is you use Claude maybe as a planning agent, either with a more complicated task, but then when you want to implement things, Mellum will be faster and much, much cheaper to implement the plans that Claude comes up with. And this is what this course will teach, so I would say keep an eye on our website.

Q: So which means that Mellum would be like a separate like software from the Claude Code, so if I run Mellum that means I don't have to like run Claude Code anymore?

### 子智能体、本地运行与结尾

**29:13**

A: So you're not the bottleneck between passing prompts from Claude to Mellum, and this is what we're working on, so having Claude and having Mellum as a sub-agent for Claude, essentially.

Q: Will there be like a token limit, and why and why not?

A: I mean, if you run it locally there is no limit, so that's the cool thing about this model, because if you have 16 gigs of RAM, it will run on your laptop.[^hardware] If you want to try downloading the model, there's the QR code, and the model weights are available for download. We released our latest agentic model last week, and it will perform the best.[^release] With that, I want to say thank you very much for listening. Please again do not hesitate to ask questions. I have all of the best minds of our training team here; they went into this a little deeper than I did, so they can definitely answer any kind of technical stuff. Thank you for listening.

### 工具调用能力从何而来

**31:23**

Yeah, thanks for the question. So we have actually a lot of tool calling data sets. They were — Nemotron lab, you know, NVIDIA released a lot of open source data sets with the tool calls that can be verified programmatically, so to speak.[^judge] Basic reinforcement learning with these data sets, also we generated some synthetic data. And of course, tool calling abilities also come from SWE style of learning. Also, we have a bunch of data sets for SWE tasks. The model generates something in sandboxes, and in these sandboxes, it uses tools as well.

### 工具调用与推理轨迹数据问答

**32:21**

So, yeah, tool calling abilities, they emerge from there. And it's backed up by benchmarks. We have one for tool calling specifically. [^benchmark]

I want to ask, in your presentation, you say in the SFT, you... I wanted to know your cold-starting of the, sorry, my English is very poor, of the reasoning traces. You say you use the open source data from Hugging Face, but I know the Hugging Face data somehow lacks of reasoning and good examples. So I want to know, have you discovered any model to cold-start your reasoning? And have you considered to use the data from some traces in JetBrains IDE to build a better reasoning?

### SFT 中推理轨迹的使用

**33:59**

So, the question is, did we use reasoning traces in the SFT? Yes, from Hugging Face data. Did I get it right? Probably I can answer your question. As I understood your question, most of the Hugging Face datasets that we were using already included reasoning traces, so we do not need to regenerate them using some open source model or our own model, but also as Aral mentioned, we generated our own synthetic data, we took some comprehensive tasks and generated reasoning traces and real answers using some big open source models such as Kimi, DeepSeek or some other top-top models. Did I understand your question?

### 开发过程中 AI 工具的使用

**34:55**

In developing this model, so how much AI have you used in this, in developing this model? Yeah, okay, so as Aral said, at the time we started we were a team of six people, so six people is not a lot. So we must rely on AI tools heavily. This is not a secret. So we kind of used agentic development a lot. Yeah, and of course all the modern tools were used and we can share some experience if you'd like how we use them. So for example, for example, it's not possible, of course, to read all the AI code that was produced because it's a massive amount of code. The main goal was to build layers across the stack that let us observe and verify the generated code so you kind of can validate yourself what the AI has produced. And look, for example, you have some kind of a fancy logging system that you can scan yourself with your eyes. You have a lot of logs. You can see actually what model is doing right now, whether the checks are correct and everything like this. And also, I try to kind of move a little bit higher and think in blocks. You know, just obviously Claude Code can write HTTP servers better than I can and can do Kubernetes better than I can. Why would I do it myself if I can kind of outline what I want, then verify it somehow using these observability tools. And yeah, what you need to do, they like to overcomplicate a lot, agents, I mean. And you just need to kind of keep it simple because most probably all this AI slop is not needed. You most probably need something simple that works for you in this case. And we'll just try to keep it simple in concept.

### 用 AI 生成假设与改进训练

**37:40**

I also wanted to add, while we were preparing our newest release, we used AI a lot for generating different hypotheses, how to improve every stage of our pipeline. For example, I can say from my side, when I was trying to improve our current reinforcement learning training, I was using a lot of AI to generate different hypotheses where we can improve. For example, you can give plots, for example, reward plot, loss plot, entropy plot, and ask, for example, Claude Code, Codex, or some other model, what can we improve? How can we change hyperparameters? How can you please analyze all the newest papers to make our current training better? And this is kind of, you just can feed all of your data, all of your trainings to some AI model and try to improve it that way.

I also want to know maybe the boundary of current AI tools in developing those models. So there must be some problems that AI can't solve. So our humans have to use our own intelligence to guide them.

### 资源约束与 AI 的局限

**39:05**

So what kind of those problems that you meet in the model? We have limited resources, and you need to set boundaries for the AI.[^resources] You can do kind of other research, whatever, but it doesn't really account for your specific constraints. Other teams may have the resources to run many more research experiments. But in our case, it was really hard to direct an agent into the right direction and to actually kind of communicate what constraints we have because usually it wants to run, I don't know, 50 or 60 runs.

### AI 模型的局限与观众看法

**40:22**

Sorry, I was very slow. I'd like to add one more thing that, in my opinion, the biggest constraint of current AI models is that they can't really develop something new. They can summarize your newest papers or something, they can try to tune hyper-parameters to check your data sets, check are there some obvious errors. But still, they can't make whole research pipeline, they can't do people's work right now just because people's job is to find out something new, find out what is the direction of development of our current model. So, in my opinion, this is one of the biggest constraints of AI models right now.

### AI 作为学习工具

**41:09**

Yeah, I basically agree with that. I think, obviously, you're still keeping it in a loop. I don't know if I fully agree with your statement that humans cannot be replaced by AI, to be honest, but let's assume that even that is true. I still do not see it as a reason to not use AI. I use AI to study a lot, and I think even so, say that a human in the loop is needed, you can still use AI to learn things a lot more efficiently than you would by manually having to scan through textbooks, manually read the entirety of 500 papers. But AI can help you distill that knowledge and go much more concise and easier for you to understand the format.

### 代码补全模型与通用模型的区别

**41:56**

[Listener:] You have a general model used for agentic coding. What about a specific code-completion model that can be integrated into IDEs? What's the difference between training a code-completion model and a general model?

[Speaker:] Okay, so we do actually have a coding model in the IDE for completion. It's what I presented here last year, Mellum 1.[^mellum1] The key difference is you can actually use Mellum 2 for this as well. I recommend you use the base model because it's been trained specifically on the format it expects to see for code completion tasks.

### Agent 训练与代码补全训练

**43:16**

The key difference with agentic training is that you need to do this fine-tuning and reasoning stages essentially where it learns to think before it answers and it learns to call tools. Does that answer your question?

### 延迟与即时响应

**43:30**

And so if you want to train a code completion model, like in the post-training stages, you don't have to teach it to reason for very long because the users are expecting a very instant response.

[Listener:] Is that true?

[Speaker:] Exactly. So you really care about latency there as well, like you ideally want the model to generate a completion in less than 100 milliseconds, including the round trip time to the server. So, yeah, exactly. You don't have the budget to fix that.

[Listener:] Okay, thank you.

### JetBrains IDE 免费代码补全

**44:03**

And actually, I should add, I think it is supported for free now, right? So if you have a classroom license for any kind of JetBrains IDE, you should be having free code completion powered by Mellum, as well as next-edit suggestion.

### 关于耗时与 token 消耗的提问

**44:20**

[Listener:] Superb, right? How can one minimize the time consumed by an engineering task?[^timequestion]

[Speaker:] Let me ask one more question. Do you mean when you're using a model or when you're training a model?

### 用强化学习奖励简短回答

**45:07**

The answer is that during reinforcement training, they are urging our model to generate correct, but as short as possible answers. So we are, I guess, we are giving bigger reward to the answers that are correct, but at the same time, they are shorter. So we are fixing this problem using our reinforcement training model.

### harness 与推理端的控制

**45:50**

For users, this is more of a harness problem: how you manage the conversation, agents and context window so the main agent does not get overwhelmed. Compaction and sub-agents also have to be accounted for in reinforcement learning.[^harness]

From the inference side, you can also say in your prompt that you want to solve the problem quickly, rather than conduct a large investigation that spends many tokens and a lot of time.

### thinking effort 与 thinking traces

**46:52**

There is also thinking effort, which needs to be accounted for in reinforcement learning. The speaker also discusses handling thinking traces.[^traces]

### 数据混合比例与训练顺序

**47:37**

[Listener:] About the mixture of data, I've heard about that maybe the mixture of data is very important. Maybe you have different percentage of data, like the percentage of math, code and web are very important, so can you show me more details about it? And have you done ever experiment of those kind of things?

[Speaker:] So, yeah, you're right. The data mixture matters a lot. In our ablations, what we found is that it's better to first train on more general data, the easier data to understand.

### 数据混合与训练策略

**48:23**

And later, while the model already knows basic stuff, you introduce more harder data, and you basically train on higher quality data that you repeat multiple times in order to get the best model for the data that you have. Now, how we found this exact mixture, actually it was mostly research, and the exact way we arrived at the numbers shown here is not clear in this transcript[^mixture], but for the new model that we're training, we are developing our in-house way of finding the actual data mixture.

### 结束语

**49:43**

And that being said, let's end the lecture here. We will still be around. If you want to get our contacts or speak to us about internship opportunities or other—please reach out to us, even if it's just a random question, we will be here. And thank you very much for attending.

### 合影邀请

**50:16**

感谢大家，也感谢 JetBrains 团队。最后请大家到门口，和团队成员一起合影。谢谢大家。


## 编辑说明与尚存疑点

正文按录音顺序整理，时间戳以段落起点定位。模型校订后仅对明确错误、跨批接缝和下面列出的高风险识别损坏作了保守修正；没有据公开报告补齐听不清的专名、数字或投影片内容。原始识别、复识别、模型返回和修改记录保留在本地，未随网站公开。

英文段落中，讲者对技术和行业的概括、实验结果与估计比例仍是讲者本人的表述。例如 GRPO 的简化解释和所提年份未按教科书改写；前文的 “8,000 words” 与后文的 “8,000 tokens” 也未静默统一单位。

[^olmo]: 约 17:26。可确认 Allen AI、OLMo，以及讲者团队发现某种尝试损害性能、因而未采用。训练阶段、长文本数据和合成任务的具体修饰语在多次识别中仍不稳定；正文没有把它们当作可靠术语。
[^loss]: 约 25:56。原识别作 “Cisco Loss”；具体名称及紧接着的机制描述未能可靠确认，未据近音猜写正式名称。
[^comparison]: 约 26:55。对比模型被识别为 “Grant 3.5”，名称不可靠；所说的 9-billion 参数量及速度比较也未经独立核实。这里仅保留讲者的比较陈述，不作为经过验证的 benchmark。
[^integration]: 约 28:13。Claude Code 与 Mellum 接入讨论中的一个中间短语识别损坏；没有补写为具体训练或接入方案。正文保留讲者确认“可以做到”和后续规划 agent / 实现 agent 的分工。
[^hardware]: 约 29:30。16GB 及笔记本运行是讲者的现场表述；未能确认指系统内存还是显存，也没有提供量化方式、上下文长度或硬件型号，不能据此作硬件保证。“本地没有 token 限制”亦不等于没有上下文或资源上限。
[^release]: 约 30:00。下载平台与版本的原识别短语包含 “2.1”，完整名称不稳定；未猜写下载地址。“上周”“本月稍晚”等均相对于未确认日期的讲座现场，而不是本站发布日期。
[^judge]: 约 31:23。可确认数据中的工具调用可以程序化验证；与 judge 有关的后半句不完整，未推导为不需要人工标注或不需要任何评判模型。
[^benchmark]: 约 32:31。工具调用基准的缩写在两次识别中不一致，可能接近 BFCL，但未可靠确认，正文未指定其正式名称。
[^resources]: 约 39:13。该段明确在讲资源限制和向 agent 传达约束；某些形容词及比较对象的名称未恢复，没有写成“没有互联网”等具体限制。
[^mellum1]: 约 42:58。Mellum 1 后的参数量缺失，未用公开资料补数。
[^timequestion]: 约 44:20。听众问题中工程任务后半句不能可靠还原；保留关于时间消耗的核心问题，不补写父任务、反馈机制等细节。
[^harness]: 约 45:50。可确认讨论 harness、上下文、compaction、sub-agents 与训练配合，但 reduce / introduce 等动词不稳定，未据此写成“应减少 compaction 或子智能体”的建议。
[^traces]: 约 46:52。thinking effort 与强化学习的关系可以确认，但具体过程及处理 thinking traces 的动词（strip / scrape）无法区分。正文没有将其改写为采集第三方思维链的方法。
[^mixture]: 约 48:42。讲者明确说配比主要来自研究探索，并在为新模型开发内部配比方法；关于旧配比数字如何得到的一个短语损坏，未推定具体搜索或消融程序。

---

录音及讲述内容的权利归原讲者和相关权利人；本站仅作学习整理。若发现转录错误或需要移除内容，可通过本站仓库提交 issue。
