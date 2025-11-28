# Needle in a Haystack 数据集转换工具

这个工具用于将 Needle in a Haystack 测试转换为标准的数据集格式，输出为 `(prompt, text, label)` 元组的列表。

## 什么是 Needle in a Haystack 测试？

Needle in a Haystack 是一个用于测试大语言模型长上下文检索能力的测试方法：

1. **Needle（针）**: 一个需要检索的事实或陈述
2. **Haystack（干草堆）**: 长文本背景（通常来自 Paul Graham 的文章）
3. **测试过程**: 将 needle 插入到 haystack 的不同位置（深度），并要求模型检索这个信息

这个测试通过改变：
- **上下文长度（Context Length）**: 文本的总长度
- **文档深度（Document Depth）**: needle 在文本中的位置（百分比）

来全面评估模型的长文本理解和检索能力。

## 数据格式

转换后的数据集是一个列表，每个元素是一个包含三个字符串的元组：

```python
(prompt, text, label)
```

- **prompt**: 检索问题（例如："What is the best thing to do in San Francisco?"）
- **text**: 完整的上下文文本，包含插入的 needle
- **label**: needle 的内容（正确答案）

## 安装依赖

建议安装 `tiktoken` 以获得更准确的 token 计数：

```bash
pip install tiktoken
```

如果没有安装 tiktoken，脚本会使用简化的字符计数方法（可能不够精确）。

## 使用方法

### 1. 基本使用 - 使用默认配置

```python
from needle_in_a_haystack import convert_needle_haystack_dataset

# 使用默认配置生成数据集
dataset = convert_needle_haystack_dataset(
    context_lengths=[2000, 4000, 8000],
    document_depth_percents=[0, 25, 50, 75, 100],
)

# 访问数据
for prompt, text, label in dataset:
    print(f"问题: {prompt}")
    print(f"文本长度: {len(text)}")
    print(f"答案: {label}")
    print("-" * 50)
```

### 2. 自定义 Needle 和问题

```python
from needle_in_a_haystack import convert_needle_haystack_dataset

dataset = convert_needle_haystack_dataset(
    needle="The secret password is: OpenSesame123",
    retrieval_question="What is the secret password?",
    context_lengths=[2000, 4000, 8000],
    document_depth_percents=[0, 50, 100],
)
```

### 3. 保存和加载数据集

```python
from needle_in_a_haystack import (
    convert_needle_haystack_dataset,
    save_dataset,
    load_dataset
)

# 生成数据集
dataset = convert_needle_haystack_dataset(
    context_lengths=[2000, 4000],
    document_depth_percents=[0, 50, 100],
)

# 保存到文件
save_dataset(dataset, "my_dataset.json")

# 从文件加载
loaded_dataset = load_dataset("my_dataset.json")
```

### 4. 使用类进行更多控制

```python
from needle_in_a_haystack import NeedleHaystackDatasetConverter

converter = NeedleHaystackDatasetConverter(
    needle="Custom needle text",
    haystack_dir="PaulGrahamEssays",
    retrieval_question="What is the custom fact?",
    context_lengths=[1000, 2000, 4000, 8000],
    document_depth_percents=[0, 10, 25, 50, 75, 90, 100],
    final_context_length_buffer=200,
    model_name="gpt-3.5-turbo"
)

dataset = converter.convert()
```

## 参数说明

### `convert_needle_haystack_dataset()` 函数参数

- **needle** (str): 需要检索的文本（作为 label）
  - 默认值: "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."

- **haystack_dir** (str): 包含背景文本的目录
  - 默认值: "PaulGrahamEssays"
  - 这个目录应该在 `LLMTest_NeedleInAHaystack/needlehaystack/` 下

- **retrieval_question** (str): 检索问题（作为 prompt）
  - 默认值: "What is the best thing to do in San Francisco?"

- **context_lengths** (List[int]): 上下文长度列表（以 token 为单位）
  - 默认值: `[2000, 4000, 8000, 16000, 32000, 64000]`
  - 建议从小值开始测试，大值会生成非常长的文本

- **document_depth_percents** (List[int]): 文档深度百分比列表
  - 默认值: `[0, 10, 25, 50, 75, 90, 100]`
  - 0 表示在开头，100 表示在末尾

- **final_context_length_buffer** (int): 上下文长度缓冲区
  - 默认值: 200
  - 用于为系统消息和输出预留空间

## 运行示例

直接运行脚本可以看到完整的示例：

```bash
cd /home/user/Gurthang/LMPG
python3 tasks/retrieval/needle_in_a_haystack.py
```

这将会：
1. 生成默认配置的数据集
2. 将数据集保存到文件
3. 生成自定义配置的数据集
4. 从文件加载数据集

## 数据集大小

生成的数据集大小 = `len(context_lengths) × len(document_depth_percents)`

例如：
- `context_lengths=[2000, 4000, 8000]` (3个)
- `document_depth_percents=[0, 25, 50, 75, 100]` (5个)
- 总共生成：3 × 5 = 15 个样本

## 注意事项

1. **Token 计数**: 使用 tiktoken 进行精确的 token 计数，如果没有安装会使用近似方法
2. **文本长度**: 较大的 context_length 会生成非常长的文本，注意存储空间
3. **生成时间**: 生成大量样本可能需要一些时间，因为需要读取和处理文本文件
4. **Needle 插入**: Needle 会在句子边界（句号后）插入，以保持文本的自然性

## 示例输出

```python
# 第一个样本
prompt = "What is the best thing to do in San Francisco?"

text = """
The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.
December 2014If the world were static, we could have monotonically increasing
confidence in our beliefs...
[很长的文本，包含多篇 Paul Graham 的文章]
"""

label = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
```

## 原始数据集来源

这个转换工具基于 [LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) 项目。

原始项目用于直接测试大语言模型的长上下文能力，而这个工具将其转换为标准的数据集格式，方便用于训练、评估或其他研究目的。

## 许可证

遵循原始 LLMTest_NeedleInAHaystack 项目的 MIT 许可证。

