"""
数据集合并脚本
将增强生成的新QA对合并到原始数据集中
"""

import json
from pathlib import Path
from typing import List, Dict


class DatasetMerger:
    def __init__(self):
        """初始化数据集合并器"""
        pass
    
    def load_json(self, file_path: str) -> Dict:
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_json(self, data: List[Dict], file_path: str):
        """保存JSON文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def merge_datasets(self, 
                      original_file: str, 
                      augmented_file: str, 
                      output_file: str,
                      backup: bool = True):
        """
        合并数据集
        
        Args:
            original_file: 原始数据集文件路径
            augmented_file: 增强数据集文件路径
            output_file: 输出文件路径
            backup: 是否备份原始文件
        """
        print(f"📂 正在加载原始数据集: {original_file}")
        original_data = self.load_json(original_file)
        
        print(f"📂 正在加载增强数据集: {augmented_file}")
        augmented_data = self.load_json(augmented_file)
        
        # 获取原始QA对列表
        if isinstance(original_data, list):
            original_qa_list = original_data
        else:
            original_qa_list = original_data.get('entries', [])
        
        # 获取新生成的QA对
        new_qa_pairs = augmented_data.get('new_qa_pairs', [])
        
        # 获取增强结果，用于过滤不可接受的问题
        augmented_results = augmented_data.get('augmented_results', [])
        
        print(f"\n📊 数据统计:")
        print(f"   原始数据集: {len(original_qa_list)} 条")
        print(f"   新生成QA对: {len(new_qa_pairs)} 条")
        
        # 备份原始文件
        if backup:
            backup_file = str(Path(original_file).with_suffix('.backup.json'))
            print(f"\n💾 备份原始文件到: {backup_file}")
            self.save_json(original_qa_list, backup_file)
        
        # 合并数据集
        merged_data = []
        
        # 1. 添加原始数据中可接受的问题
        acceptable_count = 0
        unacceptable_count = 0
        
        # 创建问题到评估结果的映射
        question_to_eval = {}
        for result in augmented_results:
            question = result.get('question', '')
            question_to_eval[question] = result
        
        for qa in original_qa_list:
            question = qa.get('question', '')
            
            # 检查该问题的可接受性
            if question in question_to_eval:
                eval_result = question_to_eval[question]
                if eval_result.get('is_acceptable', True):
                    merged_data.append(qa)
                    acceptable_count += 1
                else:
                    unacceptable_count += 1
            else:
                # 如果没有评估结果，默认保留
                merged_data.append(qa)
                acceptable_count += 1
        
        print(f"   保留可接受问题: {acceptable_count} 条")
        print(f"   过滤不可接受问题: {unacceptable_count} 条")
        
        # 2. 添加新生成的QA对
        for new_qa in new_qa_pairs:
            # 转换为标准格式（移除source字段）
            qa_item = {
                "question": new_qa.get('question', ''),
                "answer": new_qa.get('answer', '')
            }
            merged_data.append(qa_item)
        
        print(f"   添加新生成QA对: {len(new_qa_pairs)} 条")
        print(f"\n✅ 合并后总数: {len(merged_data)} 条")
        
        # 保存合并后的数据集
        print(f"\n💾 保存合并数据集到: {output_file}")
        self.save_json(merged_data, output_file)
        
        # 打印详细统计
        print("\n" + "="*50)
        print("📊 合并统计信息:")
        print(f"   原始数据集: {len(original_qa_list)} 条")
        print(f"   - 保留: {acceptable_count} 条")
        print(f"   - 过滤: {unacceptable_count} 条")
        print(f"   新增QA对: {len(new_qa_pairs)} 条")
        print(f"   合并后总数: {len(merged_data)} 条")
        print(f"   数据增长: +{len(merged_data) - len(original_qa_list)} 条 "
              f"({(len(merged_data) - len(original_qa_list)) / len(original_qa_list) * 100:.1f}%)")
        print("="*50)
        
        return merged_data
    
    def deduplicate(self, qa_list: List[Dict]) -> List[Dict]:
        """
        去重（基于问题）
        
        Args:
            qa_list: QA对列表
            
        Returns:
            去重后的QA对列表
        """
        seen_questions = set()
        deduplicated = []
        duplicates = 0
        
        for qa in qa_list:
            question = qa.get('question', '').strip()
            if question and question not in seen_questions:
                seen_questions.add(question)
                deduplicated.append(qa)
            else:
                duplicates += 1
        
        print(f"\n🔍 去重完成: 移除 {duplicates} 条重复数据")
        return deduplicated


def main():
    """主函数"""
    # 配置路径
    data_dir = Path(__file__).parent.parent / "artifacts" / "dataset"
    original_file = data_dir / "qa_dataset.json"
    augmented_file = data_dir / "qa_dataset_augmented.json"
    output_file = data_dir / "qa_dataset.json"  # 覆盖原文件（已备份）
    
    # 检查文件是否存在
    if not original_file.exists():
        print(f"❌ 原始数据集文件不存在: {original_file}")
        return
    
    if not augmented_file.exists():
        print(f"❌ 增强数据集文件不存在: {augmented_file}")
        print("请先运行 augment_dataset.py 生成增强数据集")
        return
    
    # 初始化合并器
    merger = DatasetMerger()
    
    # 合并数据集
    merged_data = merger.merge_datasets(
        original_file=str(original_file),
        augmented_file=str(augmented_file),
        output_file=str(output_file),
        backup=True  # 自动备份原始文件
    )
    
    # 可选：去重
    print("\n🔍 检查重复数据...")
    deduplicated_data = merger.deduplicate(merged_data)
    
    if len(deduplicated_data) < len(merged_data):
        print(f"💾 保存去重后的数据集")
        merger.save_json(deduplicated_data, str(output_file))
    
    print("\n✅ 数据集合并完成!")
    print(f"📁 输出文件: {output_file}")
    print(f"📁 备份文件: {output_file.with_suffix('.backup.json')}")


if __name__ == "__main__":
    main()
