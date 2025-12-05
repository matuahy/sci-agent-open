## 学习总结
### langgraph
* 创建智能体的框架选取langgraph
  * 这是创建的示例[https://langgraph.com.cn/tutorials/multi_agent/agent_supervisor.1.html#set-up-agent-communication]
  * what is React_agent?
    * ReAct Agent (Reasoning + Acting Agent)的核心思想是让大模型在执行任务时，先“思考”(Reasoning),再“行动”(Acting),并根据行动的结果进行下一步的思考。
    * Thought -> Action -> Observation -> Reasoning
  * what is 提示词prompt?
    * CRISPE 框架
      * Persona (角色/人设)
      * Context (背景上下文)
      * Instruction (指令/任务)
      * Constraints (约束/防守)
      * Few-Shot Examples (少样本示例)
      * Output Format (输出格式)
    * Dynamic Prompting (动态提示词)
  * langgraph创建工作流的原理是什么？
### RAG

### 构建一个项目所需的东西
* 本地部署 
  * 模型数据库[https://huggingface.co/]
  * 用huggingface镜像下载模型到本地 [https://hf-mirror.com/]
  * 底层计算引擎与硬件抽象层：PyTorch (torch)
    * torch 扮演深度学习框架（Deep Learning Framework）的角色,它提供了底层的计算图（Computational Graph）执行能力和硬件抽象（Hardware Abstraction）。
    * 深度学习模型的本质是海量的矩阵运算（Matrix Multiplication）和非线性变换。Python 原生的 list 数据结构无法高效处理这种高维数据。torch 提供了 Tensor（张量）数据结构，这是多维矩阵在内存中的封装。它底层使用 C++/CUDA 编写，能够利用 SIMD（单指令多数据流）指令集进行极速运算。
    * torch 负责管理**主存（RAM）与显存（VRAM）**之间的数据总线（PCIe）通信。
    * 虽然是推理模式（eval()），PyTorch 依然需要构建（或复用）前向传播的计算图，定义数据如何在各个神经网络层之间流动。
  * 模型架构实现层：Transformers (transformers)
    * transformers 库是应用层抽象（Application Abstraction Layer），它实现了具体的神经网络拓扑结构（Neural Network Topology）和预处理流水线。
  * 对于LLM(生成模型，如Llama-3-8B)，这是一个自回归（Auto-regressive）过程，一个词一个词往外蹦，极度依赖 KV Cache。
    * 在传统的 Transformers 推理中（比如生成文本），模型需要记住之前生成过的 token 的计算状态，这叫做 KV Cache (Key-Value Cache)
    * 部署用vLLM，vLLM 借鉴了操作系统（OS）管理内存的思路。
      * vLLM引入了 PagedAttention 算法。它把 KV Cache 切分成很多小块（Block）。
        * token 的数据可以存在显存的任意角落，不需要连续。
        * 由 vLLM 维护一张“表”来记录哪个 token 的数据在哪里。
* API调用
  * 用阿里百炼获取常用LLM的API
  * https://www.aliyun.com/product/bailian
* GPU
  * AutoDL算力云    https://autodl.com/home
* github