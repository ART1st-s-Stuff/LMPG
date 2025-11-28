"""
转换Needle in a Haystack数据集
输出格式：列表，每个元素是(prompt, text, label)的元组
"""
import os
import glob
import json
import random
import numpy as np
from typing import List, Tuple, Dict, Any

try:
    import tiktoken
except ImportError:
    print("警告: tiktoken未安装，将使用简单的字符计数方法")
    tiktoken = None


class NeedleHaystackDatasetConverter:
    """
    Needle in a Haystack数据集转换器
    """
    
    def __init__(
        self,
        needle: str = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
        haystack_dir: str = "PaulGrahamEssays",
        retrieval_question: str = "What is the best thing to do in San Francisco?",
        context_lengths: List[int] = None,
        document_depth_percents: List[int] = None,
        final_context_length_buffer: int = 200,
        model_name: str = "gpt-3.5-turbo",
    ):
        """
        初始化转换器
        
        Args:
            needle: 需要检索的文本（作为label）
            haystack_dir: 包含背景文本的目录
            retrieval_question: 检索问题（作为prompt）
            context_lengths: 上下文长度列表
            document_depth_percents: 文档深度百分比列表
            final_context_length_buffer: 上下文长度缓冲区
            model_name: 用于tokenization的模型名称
        """
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.final_context_length_buffer = final_context_length_buffer
        self.model_name = model_name
        
        # 设置默认的context_lengths
        if context_lengths is None:
            self.context_lengths = [2000, 4000, 8000, 16000, 32000, 64000]
        else:
            self.context_lengths = context_lengths
        
        # 设置默认的document_depth_percents
        if document_depth_percents is None:
            self.document_depth_percents = [0, 10, 25, 50, 75, 90, 100]
        else:
            self.document_depth_percents = document_depth_percents
        
        # 初始化tokenizer
        if tiktoken is not None:
            try:
                self.encoding = tiktoken.encoding_for_model(model_name)
            except:
                # 如果模型不支持，使用默认编码
                self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoding = None
        
    def read_context_files(self) -> str:
        """读取haystack目录中的所有文本文件"""
        context = ""
        max_context_length = max(self.context_lengths)
        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'LLMTest_NeedleInAHaystack',
            'needlehaystack',
            self.haystack_dir
        )
        
        # 读取文件直到达到最大上下文长度
        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(os.path.join(base_dir, "*.txt")):
                with open(file, 'r', encoding='utf-8') as f:
                    context += f.read()
                if self.get_context_length_in_tokens(context) >= max_context_length:
                    break
        
        return context
    
    def encode_text_to_tokens(self, text: str) -> List[int]:
        """将文本编码为tokens"""
        if self.encoding is not None:
            return self.encoding.encode(text)
        else:
            # 如果没有tiktoken，使用简单估计：约4个字符=1个token
            return list(range(len(text) // 4))
    
    def decode_tokens(self, tokens: List[int]) -> str:
        """将tokens解码为文本"""
        if self.encoding is not None:
            return self.encoding.decode(tokens)
        else:
            # 简单fallback - 这在没有tiktoken时不会工作得很好
            # 但至少可以让代码运行
            raise NotImplementedError("需要安装tiktoken才能正确解码tokens")
    
    def get_context_length_in_tokens(self, context: str) -> int:
        """获取文本的token长度"""
        return len(self.encode_text_to_tokens(context))
    
    def encode_and_trim(self, context: str, context_length: int) -> str:
        """将context编码为tokens并修剪到指定长度"""
        tokens = self.encode_text_to_tokens(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens[:context_length])
        return context
    
    def insert_needle(self, context: str, depth_percent: float, context_length: int) -> str:
        """在指定深度位置插入needle"""
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)
        
        # 减去缓冲区以留出输出空间
        context_length -= self.final_context_length_buffer
        
        # 如果context + needle长度超过context_length，则修剪context
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]
        
        if depth_percent == 100:
            # 如果深度百分比是100，将needle放在末尾
            tokens_new_context = tokens_context + tokens_needle
        else:
            # 计算插入位置
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            
            # 在句子边界处插入
            tokens_new_context = tokens_context[:insertion_point]
            
            # 找到句号token
            period_tokens = self.encode_text_to_tokens('.')
            
            # 向后查找直到找到句号
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]
            
            # 插入needle并添加剩余的context
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]
        
        # 转换回字符串
        new_context = self.decode_tokens(tokens_new_context)
        return new_context
    
    def generate_context(self, context_length: int, depth_percent: float) -> str:
        """生成指定长度和深度的context"""
        # 读取背景文本
        context = self.read_context_files()
        
        # 修剪到指定长度
        context = self.encode_and_trim(context, context_length)
        
        # 插入needle
        context = self.insert_needle(context, depth_percent, context_length)
        
        return context
    
    def convert(self) -> List[Tuple[str, str, str]]:
        """
        转换数据集
        
        Returns:
            List[Tuple[str, str, str]]: 列表，每个元素是(prompt, text, label)的元组
                - prompt: 检索问题
                - text: 完整的包含needle的context
                - label: needle文本（正确答案）
        """
        dataset = []
        
        print(f"开始转换Needle in a Haystack数据集...")
        print(f"Context长度数量: {len(self.context_lengths)}")
        print(f"深度百分比数量: {len(self.document_depth_percents)}")
        print(f"总样本数: {len(self.context_lengths) * len(self.document_depth_percents)}")
        
        for i, context_length in enumerate(self.context_lengths):
            for j, depth_percent in enumerate(self.document_depth_percents):
                print(f"处理 [{i+1}/{len(self.context_lengths)}][{j+1}/{len(self.document_depth_percents)}] "
                      f"Context长度={context_length}, 深度={depth_percent}%")
                
                try:
                    # 生成context
                    context = self.generate_context(context_length, depth_percent)
                    
                    # 创建元组：(prompt, text, label)
                    data_tuple = (
                        self.retrieval_question,  # prompt
                        context,                   # text
                        self.needle.strip()       # label
                    )
                    
                    dataset.append(data_tuple)
                    
                except Exception as e:
                    print(f"错误: 处理context_length={context_length}, depth={depth_percent}时出错: {e}")
                    continue
        
        print(f"\n转换完成！总共生成了 {len(dataset)} 个样本")
        return dataset


def convert_needle_haystack_dataset(
    needle: str = None,
    haystack_dir: str = "PaulGrahamEssays",
    retrieval_question: str = None,
    context_lengths: List[int] = None,
    document_depth_percents: List[int] = None,
    final_context_length_buffer: int = 200,
) -> List[Tuple[str, str, str]]:
    """
    转换Needle in a Haystack数据集的便捷函数
    
    Args:
        needle: 需要检索的文本（作为label）
        haystack_dir: 包含背景文本的目录
        retrieval_question: 检索问题（作为prompt）
        context_lengths: 上下文长度列表
        document_depth_percents: 文档深度百分比列表
        final_context_length_buffer: 上下文长度缓冲区
    
    Returns:
        List[Tuple[str, str, str]]: 列表，每个元素是(prompt, text, label)的元组
    """
    # 默认值
    if needle is None:
        needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
    
    if retrieval_question is None:
        retrieval_question = "What is the best thing to do in San Francisco?"
    
    converter = NeedleHaystackDatasetConverter(
        needle=needle,
        haystack_dir=haystack_dir,
        retrieval_question=retrieval_question,
        context_lengths=context_lengths,
        document_depth_percents=document_depth_percents,
        final_context_length_buffer=final_context_length_buffer,
    )
    
    return converter.convert()


def save_dataset(dataset: List[Tuple[str, str, str]], output_file: str):
    """
    保存数据集到JSON文件
    
    Args:
        dataset: 数据集列表
        output_file: 输出文件路径
    """
    import json
    
    # 将元组转换为字典格式以便保存
    data_dicts = []
    for prompt, text, label in dataset:
        data_dicts.append({
            "prompt": prompt,
            "text": text,
            "label": label
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_dicts, f, ensure_ascii=False, indent=2)
    
    print(f"数据集已保存到: {output_file}")


def load_dataset(input_file: str) -> List[Tuple[str, str, str]]:
    """
    从JSON文件加载数据集
    
    Args:
        input_file: 输入文件路径
    
    Returns:
        List[Tuple[str, str, str]]: 数据集列表
    """
    import json
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data_dicts = json.load(f)
    
    dataset = []
    for item in data_dicts:
        dataset.append((item["prompt"], item["text"], item["label"]))
    
    return dataset


class MultiNeedleHaystackConverter:
    """
    多Needle in a Haystack数据集转换器
    用于处理多跳推理数据集，将多个derivations随机插入到haystack中
    """
    
    def __init__(
        self,
        haystack_dir: str = "PaulGrahamEssays",
        context_lengths: List[int] = None,
        final_context_length_buffer: int = 200,
        model_name: str = "gpt-3.5-turbo",
        random_seed: int = None,
    ):
        """
        初始化多needle转换器
        
        Args:
            haystack_dir: 包含背景文本的目录
            context_lengths: 上下文长度列表
            final_context_length_buffer: 上下文长度缓冲区
            model_name: 用于tokenization的模型名称
            random_seed: 随机种子，用于控制needle插入位置的随机性
        """
        self.haystack_dir = haystack_dir
        self.final_context_length_buffer = final_context_length_buffer
        self.model_name = model_name
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # 设置默认的context_lengths
        if context_lengths is None:
            self.context_lengths = [4000, 8000, 16000]
        else:
            self.context_lengths = context_lengths
        
        # 初始化tokenizer
        if tiktoken is not None:
            try:
                self.encoding = tiktoken.encoding_for_model(model_name)
            except:
                self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoding = None
    
    def encode_text_to_tokens(self, text: str) -> List[int]:
        """将文本编码为tokens"""
        if self.encoding is not None:
            return self.encoding.encode(text)
        else:
            return list(range(len(text) // 4))
    
    def decode_tokens(self, tokens: List[int]) -> str:
        """将tokens解码为文本"""
        if self.encoding is not None:
            return self.encoding.decode(tokens)
        else:
            raise NotImplementedError("需要安装tiktoken才能正确解码tokens")
    
    def get_context_length_in_tokens(self, context: str) -> int:
        """获取文本的token长度"""
        return len(self.encode_text_to_tokens(context))
    
    def read_context_files(self) -> str:
        """读取haystack目录中的所有文本文件"""
        context = ""
        max_context_length = max(self.context_lengths)
        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'LLMTest_NeedleInAHaystack',
            'needlehaystack',
            self.haystack_dir
        )
        
        # 读取文件直到达到最大上下文长度
        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(os.path.join(base_dir, "*.txt")):
                with open(file, 'r', encoding='utf-8') as f:
                    context += f.read()
                if self.get_context_length_in_tokens(context) >= max_context_length:
                    break
        
        return context
    
    def encode_and_trim(self, context: str, context_length: int) -> str:
        """将context编码为tokens并修剪到指定长度"""
        tokens = self.encode_text_to_tokens(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens[:context_length])
        return context
    
    def insert_needles_randomly(
        self, 
        context: str, 
        needles: List[str], 
        context_length: int
    ) -> str:
        """
        将多个needles随机插入到context中
        
        Args:
            context: 原始上下文
            needles: 需要插入的needle列表（derivations）
            context_length: 目标上下文长度
        
        Returns:
            str: 插入needles后的完整上下文
        """
        tokens_context = self.encode_text_to_tokens(context)
        
        # 计算所有needles的token总长度
        all_needle_tokens = []
        for needle in needles:
            needle_with_space = "\n" + needle.strip() + "\n"
            all_needle_tokens.append(self.encode_text_to_tokens(needle_with_space))
        
        total_needle_length = sum(len(tokens) for tokens in all_needle_tokens)
        
        # 调整context长度，为needles和buffer留出空间
        adjusted_context_length = context_length - total_needle_length - self.final_context_length_buffer
        
        if adjusted_context_length < 100:
            adjusted_context_length = 100  # 最小context长度
        
        # 修剪context
        if len(tokens_context) > adjusted_context_length:
            tokens_context = tokens_context[:adjusted_context_length]
        
        # 找出所有句号位置（潜在的插入点）
        period_tokens = self.encode_text_to_tokens('.')
        sentence_boundaries = []
        for i, token in enumerate(tokens_context):
            if token in period_tokens:
                sentence_boundaries.append(i + 1)  # 在句号后插入
        
        # 如果没有足够的句子边界，创建均匀分布的插入点
        if len(sentence_boundaries) < len(needles):
            step = len(tokens_context) // (len(needles) + 1)
            sentence_boundaries = [step * (i + 1) for i in range(len(needles))]
        
        # 随机选择插入位置（确保不重复）
        if len(sentence_boundaries) >= len(needles):
            insertion_points = sorted(random.sample(sentence_boundaries, len(needles)))
        else:
            # 如果边界不够，使用均匀分布
            step = len(tokens_context) // (len(needles) + 1)
            insertion_points = [step * (i + 1) for i in range(len(needles))]
        
        # 从后向前插入needles（避免位置偏移）
        result_tokens = tokens_context.copy()
        for i in range(len(needles) - 1, -1, -1):
            insert_pos = insertion_points[i]
            # 确保插入位置在有效范围内
            if insert_pos > len(result_tokens):
                insert_pos = len(result_tokens)
            result_tokens = result_tokens[:insert_pos] + all_needle_tokens[i] + result_tokens[insert_pos:]
        
        # 转换回字符串
        new_context = self.decode_tokens(result_tokens)
        return new_context
    
    def convert_from_json_items(
        self, 
        json_items: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, str]]:
        """
        从JSON格式的数据项转换为数据集
        
        Args:
            json_items: JSON格式的数据项列表，每个项包含：
                - id: 数据ID
                - derivations: 推导步骤列表
                - question: 问题（作为prompt）
                - answer: 答案（作为label）
                - 其他字段（可选）
        
        Returns:
            List[Tuple[str, str, str]]: (prompt, text, label)元组列表
        """
        dataset = []
        
        # 读取背景文本
        print("读取背景文本...")
        haystack_base = self.read_context_files()
        
        print(f"\n开始转换多needle数据集...")
        print(f"数据项数量: {len(json_items)}")
        print(f"Context长度选项: {self.context_lengths}")
        
        total_samples = len(json_items) * len(self.context_lengths)
        current = 0
        
        for item in json_items:
            item_id = item.get("id", "unknown")
            derivations = item.get("derivations", [])
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            if not derivations or not question:
                print(f"警告: 跳过无效数据项 {item_id}")
                continue
            
            # 为每个context_length生成一个样本
            for context_length in self.context_lengths:
                current += 1
                print(f"处理 [{current}/{total_samples}] ID={item_id[:12]}..., "
                      f"Context长度={context_length}, Derivations数={len(derivations)}")
                
                try:
                    # 准备haystack
                    haystack = self.encode_and_trim(haystack_base, context_length * 2)
                    
                    # 随机插入所有derivations
                    text_with_needles = self.insert_needles_randomly(
                        haystack, 
                        derivations, 
                        context_length
                    )
                    
                    # 创建元组：(prompt, text, label)
                    data_tuple = (
                        question,           # prompt
                        text_with_needles,  # text
                        answer             # label
                    )
                    
                    dataset.append(data_tuple)
                    
                except Exception as e:
                    print(f"错误: 处理数据项 {item_id} 时出错: {e}")
                    continue
        
        print(f"\n转换完成！总共生成了 {len(dataset)} 个样本")
        return dataset
    
    def convert_from_json_file(
        self, 
        json_file: str
    ) -> List[Tuple[str, str, str]]:
        """
        从JSON文件转换为数据集
        
        Args:
            json_file: JSON文件路径
        
        Returns:
            List[Tuple[str, str, str]]: (prompt, text, label)元组列表
        """
        print(f"从文件加载数据: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            json_items = json.load(f)
        
        return self.convert_from_json_items(json_items)


def convert_multi_needle_dataset(
    json_items: List[Dict[str, Any]] = None,
    json_file: str = None,
    haystack_dir: str = "PaulGrahamEssays",
    context_lengths: List[int] = None,
    final_context_length_buffer: int = 200,
    random_seed: int = None,
) -> List[Tuple[str, str, str]]:
    """
    转换多needle数据集的便捷函数
    
    Args:
        json_items: JSON格式的数据项列表
        json_file: JSON文件路径（如果提供，会忽略json_items）
        haystack_dir: 包含背景文本的目录
        context_lengths: 上下文长度列表
        final_context_length_buffer: 上下文长度缓冲区
        random_seed: 随机种子
    
    Returns:
        List[Tuple[str, str, str]]: (prompt, text, label)元组列表
    """
    converter = MultiNeedleHaystackConverter(
        haystack_dir=haystack_dir,
        context_lengths=context_lengths,
        final_context_length_buffer=final_context_length_buffer,
        random_seed=random_seed,
    )
    
    if json_file:
        return converter.convert_from_json_file(json_file)
    elif json_items:
        return converter.convert_from_json_items(json_items)
    else:
        raise ValueError("必须提供json_items或json_file参数")


def get_dataset_tuples(
    json_file: str = None,
    json_items: List[Dict[str, Any]] = None,
    haystack_dir: str = "PaulGrahamEssays",
    context_lengths: List[int] = None,
    final_context_length_buffer: int = 200,
    random_seed: int = None,
    output_file: str = None
) -> List[Tuple[str, str, str]]:
    """
    从JSON文件或数据项生成数据集，并返回(prompt, text, label)元组列表
    
    这是一个便捷函数，整合了数据加载、转换和可选的保存功能
    
    Args:
        json_file: JSON文件路径
        json_items: JSON格式的数据项列表
        haystack_dir: 包含背景文本的目录
        context_lengths: 上下文长度列表（默认: [4000, 8000, 16000]）
        final_context_length_buffer: 上下文长度缓冲区
        random_seed: 随机种子
        output_file: 可选，如果提供则保存数据集到文件
    
    Returns:
        List[Tuple[str, str, str]]: (prompt, text, label)元组列表
    
    Example:
        >>> # 从文件获取数据集
        >>> dataset = get_dataset_tuples(
        ...     json_file="multi_needle_reasoning_en.json",
        ...     context_lengths=[4000, 8000],
        ...     random_seed=42
        ... )
        >>> 
        >>> # 使用数据集
        >>> for prompt, text, label in dataset:
        ...     print(f"Question: {prompt}")
        ...     print(f"Answer: {label}")
    """
    # 转换数据集
    dataset = convert_multi_needle_dataset(
        json_items=json_items,
        json_file=json_file,
        haystack_dir=haystack_dir,
        context_lengths=context_lengths,
        final_context_length_buffer=final_context_length_buffer,
        random_seed=random_seed,
    )
    
    # 如果提供了输出文件，保存数据集
    if output_file:
        save_dataset(dataset, output_file)
        print(f"数据集已保存到: {output_file}")
    
    return dataset


if __name__ == "__main__":
    import sys
    
    # 示例用法1: 单needle - 使用较小的context_lengths以便快速测试
    print("="*60)
    print("示例1: 单Needle - 生成默认配置的数据集")
    print("="*60)
    
    dataset = convert_needle_haystack_dataset(
        context_lengths=[2000, 4000],
        document_depth_percents=[0, 50, 100],
    )
    
    print(f"\n生成的数据集样本数: {len(dataset)}")
    print(f"\n第一个样本:")
    print(f"Prompt: {dataset[0][0]}")
    print(f"Text长度: {len(dataset[0][1])} 字符")
    print(f"Text前200字符: {dataset[0][1][:200]}...")
    print(f"Label: {dataset[0][2]}")
    
    # 示例用法2: 多needle - 从文件读取数据
    print("\n" + "="*60)
    print("示例2: 多Needle - 从文件读取多跳推理数据集")
    print("="*60)
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_dir, "multi_needle_reasoning_en.json")
    
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"警告: 文件 {json_file_path} 不存在，跳过多needle示例")
    else:
        print(f"从文件读取数据: {json_file_path}")
        
        # 读取文件以获取数据项总数
        with open(json_file_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        print(f"文件中共有 {len(all_data)} 个数据项")
        
        # 只处理前5个数据项作为测试（可以修改这个数字）
        num_samples = 1000
        print(f"处理前 {num_samples} 个数据项作为示例...")
        
        # 使用get_dataset_tuples函数获取元组列表
        multi_dataset = get_dataset_tuples(
            json_items=all_data[:num_samples],
            context_lengths=[4000],
            random_seed=42,
            output_file="multi_needle_dataset_sample.json"
        )
        
        print(f"\n生成的多needle数据集样本数: {len(multi_dataset)}")
        if len(multi_dataset) > 0:
            print(f"\n第一个多needle样本:")
            print(f"Prompt: {multi_dataset[0][0]}")
            print(f"Text长度: {len(multi_dataset[0][1])} 字符")
            print(f"Label: {multi_dataset[0][2]}")
            
            # 检查derivations是否在文本中
            text = multi_dataset[0][1]
            derivations = all_data[0]["derivations"]
            print(f"\nDerivations在文本中的位置:")
            for i, deriv in enumerate(derivations):
                pos = text.find(deriv)
                if pos != -1:
                    relative_pos = pos / len(text) * 100
                    print(f"  Derivation {i+1}: 位于 {relative_pos:.1f}% 处")
                else:
                    print(f"  Derivation {i+1}: 未找到（可能被截断）")
        
        # 示例用法3: 使用get_dataset_tuples直接从文件读取
        print("\n" + "="*60)
        print("示例3: 使用get_dataset_tuples直接从文件读取")
        print("="*60)
        
        print(f"处理前 {num_samples} 个数据项...")
        # 注意：这里我们需要先读取并切片，因为函数会读取整个文件
        # 对于大文件，建议在实际使用时处理全部数据
        dataset_from_file = get_dataset_tuples(
            json_items=all_data[:num_samples],
            context_lengths=[4000],
            random_seed=42
        )
        
        print(f"生成的数据集样本数: {len(dataset_from_file)}")
        
        print("\n提示: 要处理完整的数据集，可以使用:")
        print("  dataset = get_dataset_tuples(")
        print("      json_file='multi_needle_reasoning_en.json',")
        print("      context_lengths=[4000, 8000, 16000],")
        print("      random_seed=42,")
        print("      output_file='full_dataset.json'")
        print("  )")
    
    # 示例用法4: 保存和加载数据集
    print("\n" + "="*60)
    print("示例4: 保存和加载单needle数据集")
    print("="*60)
    
    output_file = "single_needle_dataset.json"
    save_dataset(dataset, output_file)
    
    loaded_dataset = load_dataset(output_file)
    print(f"从文件加载的单needle数据集样本数: {len(loaded_dataset)}")
    print(f"验证: 数据一致性 = {dataset[0] == loaded_dataset[0]}")

