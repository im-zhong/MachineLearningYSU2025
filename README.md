# MachineLearningYSU2025

下面给你一个10 节课的《机器学习与深度学习》实验课大纲，特别强调“经典算法 + 动手实现 + 工程能力训练”。每节课都给出：核心目标、主要内容、动手任务、配套作业。你可以根据你们实验室的风格再细化。

⸻

🧭《机器学习与深度学习》实验课（10 讲）大纲

定位：面向研究生，强调算法原理理解 + 手写实现 + 工程实践
目标：让同学们“能讲明白 + 能手写实现 + 能调参跑通 + 能用到项目里”

⸻

第 1 讲：课程介绍 & 必备工程环境

目标：统一开发环境，训练基本工程能力
内容：
 • Python 基础，Numpy 向量化
 • 虚拟环境（conda/uv）
 • PyTorch 基本张量操作
 • Jupyter Notebook / VSCode 实践方式
 • GPU 使用、CUDA、显存管理基本概念
动手任务：
 • 手写一个向量化的线性回归（不使用 sklearn）
作业：
 • 用最小二乘法手写线性回归 + 可视化 Loss 降低过程

⸻

第 2 讲：监督学习基础：线性模型与优化

目标：理解线性模型、最优化方法
内容：
 • 线性回归、逻辑回归
 • L1/L2 正则化
 • 梯度下降、SGD、Momentum、AdaGrad
动手任务：
 • 用 Numpy 手写 Logistic Regression + Binary Classification
作业：
 • 结合不同优化器对比训练曲线

⸻

第 3 讲：树模型与集成学习（强制手写）

目标：理解非线性模型、树模型的分裂思想
内容：
 • 决策树 CART
 • 随机森林原理
 • GBDT 的训练思想（不用手写 Boosting，但要理解）
动手任务：
 • 手写一个 CART 决策树（基尼/熵）
作业：
 • 用手写模型和 sklearn 对比性能

⸻

第 4 讲：聚类与降维

目标：掌握无监督学习基本工具
内容：
 • k-means
 • PCA（手推 + 手写 SVD 或协方差分解）
动手任务：
 • 手写 k-means
 • 手写 PCA 并可视化二维结果
作业：
 • 用 PCA+KMeans 对真实数据（如MNIST）进行可视化

⸻

第 5 讲：神经网络基础 & 反向传播手写

目标：理解深度学习本质
内容：
 • MLP
 • 激活函数
 • 反向传播推导
 • 自动微分的意义
动手任务：
 • 用 Numpy 手写 MLP + 反向传播
作业：
 • 对比“手写梯度” vs “PyTorch autograd” 的差异

⸻

第 6 讲：PyTorch 实战（工程能力重点）

目标：让学生具备最基本的 PyTorch 编程能力
内容：
 • Dataset / DataLoader
 • Module / Optimizer / Loss
 • GPU 训练与显存调优
 • 常见坑（梯度不清零、模式切换）
动手任务：
 • 用 PyTorch 训练一个最简单的 MLP MNIST 分类器
作业：
 • 改进网络、增加正则化、画训练曲线

⸻

第 7 讲：卷积神经网络（CNN）与图像任务

目标：掌握深度学习最经典路线
内容：
 • 卷积、池化
 • LeNet / AlexNet / VGG / ResNet 基本结构
 • BatchNorm、Dropout
动手任务：
 • 用 PyTorch 训练一个小型 CNN
作业：
 • 对比是否使用 BatchNorm/Dropout 的精度影响

⸻

第 8 讲：序列模型 RNN / LSTM / GRU

目标：理解序列建模
内容：
 • RNN 梯度消失/爆炸现象
 • LSTM 与 GRU 的结构细节
 • 文本分类 / 情感分析
动手任务：
 • 用 PyTorch 训练一个 GRU 做情感分类（IMDb / SST2）
作业：
 • 调参作业：embedding size, hidden size, seq length 等

⸻

第 9 讲：Transformer 机制 & Self-attention 手写推导

目标：理解现代深度学习的核心
内容：
 • Self-Attention
 • Multi-Head
 • Position Embedding
 • Encoder / Decoder
动手任务：
 • 用 Numpy / PyTorch 手写 Multi-head Attention（Forward）
作业：
 • 对 Attention Map 做可视化（可加 bonus）

⸻

第 10 讲：大模型微调与现代训练技巧（工程向）

目标：引导学生接触现代 LLM / Fine-tuning 系统
内容：
 • LoRA 原理（带公式）
 • QLoRA（NF4、二阶近似）
 • RAG（简单结构）
 • Prompt 工程
动手任务：
 • 使用 HuggingFace 进行一个小规模中文 LLM 的 LoRA 微调
作业：
 • 报告：不同 LoRA rank、learning rate 对性能影响

⸻

✔ 总体特点
 • 从经典 ML → DL → Transformer → LLM 微调
 • 前半段强调手写实现（算法理解）
 • 后半段强调工程能力（PyTorch 编程 + GPU + LLM）
 • 有梯度、有进阶、有实际可落地的项目能力

⸻

如果你愿意，我可以为你继续准备：
📌 每节课的详细教案
📌 每讲的代码模板 + Notebook
📌 课程考核方式与实验报告模板
📌 为学生准备的逐步提示（hint）与调试技巧

你要继续哪个部分？
