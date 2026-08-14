## 一些定义
### 多轮对话（拆分/合并）
一个多轮对话例如
system user1 assistant1 user2 assistant2 user3 assistant3
**多轮拆分** 指的是将其拆成三条训练数据
*system user1* **assistant1**
*system user1 assistant1 user2* **assistant2**
*system user1 assistant1 user2 assistant2 user3* **assistant3**
**多轮合并** 指的是将其作为一条训练数据
system user1 **assistant1** user2 **assistant2** user3 **assistant3**
*目前llamafactory不支持多轮合并的输入，verl支持但本仓暂未实现

## v1_training_llamav1
通过脚本将openai格式多轮拆分数据处理成USE_V1的llamafactory V1格式，然后走V1链路tokenize，将tokenize后的数据传进llamafactory V0的训练链路，32K上下文通过。

## v1_training_swift
目前Qwen3.5-2B模型训练速度最快，GPU上使用megatron后端支持CP，48K上下文通过。

## v1_training_verl_new
使用脚本预处理openai格式的数据，然后使用verl进行SFT训练，48K上下文通过。