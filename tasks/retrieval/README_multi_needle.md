# 多Needle Haystack数据集转换工具

这个工具扩展了原始的Needle in a Haystack测试，支持将多个推导步骤（derivations）随机插入到haystack中，特别适用于多跳推理（multi-hop reasoning）数据集的转换。

## 什么是多Needle测试？

多Needle测试是Needle in a Haystack的扩展版本，用于测试模型在长文本中检索和整合多个相关信息片段的能力：

1. **多个Needles（多个针）**: 多个需要检索的相关信息片段（derivations）
2. **随机插入**: 这些needles被随机插入到haystack的不同位置
3. **多跳推理**: 问题需要结合所有needles来回答

例如，对于问题："It Takes a Family是对1996年由谁出版的书的回应？"

需要三个derivations：
- "It Takes a Family book is a response to the 1996 book"
- "the 1996 book is It Takes a Village"
- "It Takes a Village is published by Hillary Rodham Clinton"

答案："Hillary Rodham Clinton"

## 数据格式

### 输入格式

输入数据应该是JSON格式的列表，每个元素包含：

```json
{
    "id": "唯一标识符",
    "derivations": [
        "推导步骤1",
        "推导步骤2",
        "推导步骤3"
    ],
    "derivation_count": 3,
    "question": "需要回答的问题",
    "answer": "答案",
    "type": "问题类型（可选）",
    "level": "难度级别（可选）"
}
```

### 输出格式

输出是一个列表，每个元素是 `(prompt, text, label)` 元组：

- **prompt**: question（问题）
- **text**: 包含所有随机插入的derivations的完整haystack
- **label**: answer（答案）

## 使用方法

### 1. 基本使用 - 从Python列表转换

```python
from needle_in_a_haystack import convert_multi_needle_dataset

# 准备数据
data = [
    {
        "id": "5ab3f9b95542992ade7c6f09",
        "derivations": [
            "It Takes a Family book is a response to the 1996 book",
            "the 1996 book is It Takes a Village",
            "It Takes a Village is published by Hillary Rodham Clinton"
        ],
        "derivation_count": 3,
        "question": "It Takes a Family is a response to this 1996 book that was published by who",
        "answer": "Hillary Rodham Clinton",
        "type": "bridge",
        "level": "hard"
    }
]

# 转换数据集
dataset = convert_multi_needle_dataset(
    json_items=data,
    context_lengths=[4000, 8000, 16000],
    random_seed=42
)

# 使用数据
for prompt, text, label in dataset:
    print(f"问题: {prompt}")
    print(f"答案: {label}")
    print(f"文本长度: {len(text)} 字符")
```

### 2. 从JSON文件加载

```python
from needle_in_a_haystack import convert_multi_needle_dataset

# 从文件加载并转换
dataset = convert_multi_needle_dataset(
    json_file="my_data.json",
    context_lengths=[4000, 8000],
    random_seed=42
)
```

### 3. 使用类进行更多控制

```python
from needle_in_a_haystack import MultiNeedleHaystackConverter

# 创建转换器
converter = MultiNeedleHaystackConverter(
    haystack_dir="PaulGrahamEssays",
    context_lengths=[4000, 8000, 16000],
    final_context_length_buffer=200,
    model_name="gpt-3.5-turbo",
    random_seed=42
)

# 转换数据
dataset = converter.convert_from_json_items(data)

# 或从文件转换
dataset = converter.convert_from_json_file("my_data.json")
```

### 4. 保存和加载数据集

```python
from needle_in_a_haystack import (
    convert_multi_needle_dataset,
    save_dataset,
    load_dataset
)

# 转换数据集
dataset = convert_multi_needle_dataset(
    json_items=data,
    context_lengths=[4000, 8000]
)

# 保存到文件
save_dataset(dataset, "multi_needle_dataset.json")

# 从文件加载
loaded_dataset = load_dataset("multi_needle_dataset.json")
```

## 参数说明

### `convert_multi_needle_dataset()` 函数参数

- **json_items** (List[Dict]): JSON格式的数据项列表
  - 必须包含 `derivations`（列表）和 `question`（字符串）字段
  - 建议包含 `id` 和 `answer` 字段

- **json_file** (str): JSON文件路径
  - 如果提供，会忽略 `json_items` 参数

- **haystack_dir** (str): 包含背景文本的目录
  - 默认值: "PaulGrahamEssays"

- **context_lengths** (List[int]): 上下文长度列表（以token为单位）
  - 默认值: `[4000, 8000, 16000]`
  - 为每个数据项的每个长度生成一个样本

- **final_context_length_buffer** (int): 上下文长度缓冲区
  - 默认值: 200
  - 为系统消息和输出预留空间

- **random_seed** (int): 随机种子
  - 默认值: None
  - 设置后可以保证插入位置的可重复性

## 工作原理

### 1. Needles插入策略

- **句子边界检测**: 优先在句号后插入needles，保持文本自然性
- **随机分布**: 使用随机采样确保needles分散在文本中
- **避免重叠**: 确保每个needle插入到不同位置

### 2. 空间分配

对于给定的 `context_length`：

```
可用空间 = context_length - 所有needles的长度 - buffer
```

算法会：
1. 读取并修剪haystack到适当长度
2. 找出所有句子边界（句号位置）
3. 随机选择插入点
4. 从后向前插入needles（避免位置偏移）

### 3. 数据集大小

```
总样本数 = 数据项数量 × context_lengths数量
```

例如：
- 10个数据项
- 3个context_lengths: [4000, 8000, 16000]
- 总共生成：10 × 3 = 30 个样本

## 示例输出

```python
# 输入数据
{
    "id": "example_001",
    "derivations": [
        "The Eiffel Tower is located in Paris",
        "Paris is the capital of France",
        "France is in Europe"
    ],
    "question": "The Eiffel Tower is located in the capital of which continent's country?",
    "answer": "Europe"
}

# 输出（简化版）
prompt = "The Eiffel Tower is located in the capital of which continent's country?"

text = """
[Paul Graham的文章...]
The Eiffel Tower is located in Paris
[更多文章内容...]
Paris is the capital of France
[更多文章内容...]
France is in Europe
[更多文章内容...]
"""

label = "Europe"
```

## Derivations位置验证

脚本会自动验证derivations是否被正确插入：

```python
# 检查derivations位置
text = dataset[0][1]
derivations = data[0]["derivations"]

for i, deriv in enumerate(derivations):
    pos = text.find(deriv)
    if pos != -1:
        relative_pos = pos / len(text) * 100
        print(f"Derivation {i+1}: 位于 {relative_pos:.1f}% 处")
```

示例输出：
```
Derivation 1: 位于 12.3% 处
Derivation 2: 位于 45.7% 处
Derivation 3: 位于 78.9% 处
```

## 与HotpotQA等数据集的兼容性

这个工具特别适合转换类似HotpotQA的多跳推理数据集。如果你的数据格式稍有不同，可以轻松调整：

```python
# 如果你的数据格式是：
hotpot_data = {
    "_id": "example",
    "supporting_facts": ["Fact 1", "Fact 2"],
    "question": "Question?",
    "answer": "Answer"
}

# 转换为所需格式：
converted_data = {
    "id": hotpot_data["_id"],
    "derivations": hotpot_data["supporting_facts"],
    "question": hotpot_data["question"],
    "answer": hotpot_data["answer"]
}
```

## 注意事项

1. **Token计数**: 
   - 建议安装tiktoken以获得准确的token计数
   - `pip install tiktoken`

2. **内存使用**: 
   - 较大的context_length会生成很长的文本
   - 注意内存和存储空间

3. **随机性控制**:
   - 使用`random_seed`参数确保结果可重复
   - 不同的seed会产生不同的插入位置

4. **Derivations数量**:
   - 支持任意数量的derivations
   - 太多derivations可能导致文本过于拥挤

5. **文本质量**:
   - Derivations会在句子边界插入
   - 如果haystack没有足够的句子边界，会使用均匀分布

## 完整示例

```python
from needle_in_a_haystack import convert_multi_needle_dataset, save_dataset

# 准备多跳推理数据
multi_hop_data = [
    {
        "id": "example_001",
        "derivations": [
            "Apple Inc. was founded by Steve Jobs",
            "Steve Jobs was born in San Francisco",
            "San Francisco is in California"
        ],
        "question": "The founder of Apple Inc. was born in which state?",
        "answer": "California"
    },
    {
        "id": "example_002",
        "derivations": [
            "The Great Wall was built in China",
            "China is the most populous country",
            "The most populous country is in Asia"
        ],
        "question": "The Great Wall was built in a country located in which continent?",
        "answer": "Asia"
    }
]

# 转换数据集
dataset = convert_multi_needle_dataset(
    json_items=multi_hop_data,
    context_lengths=[4000, 8000, 16000],
    random_seed=42
)

# 保存数据集
save_dataset(dataset, "my_multi_hop_dataset.json")

print(f"生成了 {len(dataset)} 个样本")

# 查看第一个样本
prompt, text, label = dataset[0]
print(f"\n问题: {prompt}")
print(f"答案: {label}")
print(f"文本长度: {len(text)} 字符")

# 验证derivations位置
for i, deriv in enumerate(multi_hop_data[0]["derivations"]):
    pos = text.find(deriv)
    if pos != -1:
        print(f"Derivation {i+1} 位于: {pos/len(text)*100:.1f}%")
```

## 许可证

遵循原始 LLMTest_NeedleInAHaystack 项目的 MIT 许可证。

