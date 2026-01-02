import json

# 读取jsonl文件并提取所有entries中的qa_pairs
input_path = "train_dataset.jsonl"
output_path = "qa_dataset.json"
qa_pairs = []

with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    entries = data.get('entries', [])
    for entry in entries:
        for qa in entry.get('qa_pairs', []):
            question = qa.get('question')
            answer = qa.get('content')
            if question and answer:
                qa_pairs.append({'question': question, 'answer': answer})

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

print(f"已提取{len(qa_pairs)}条问答对，保存至{output_path}")
