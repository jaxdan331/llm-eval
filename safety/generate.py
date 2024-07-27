import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# 设置 device 为 cuda:0 或者 cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate(
    model_path: str = 'Qwen/Qwen2-0.5B-Instruct',
    data_path: str = 'data/seval.jsonl',
    batch_size: int = 4,
    out_file: str = 'qwen_results.jsonl'
):
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='left'
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()

    # 加载数据集
    data = [json.loads(line) for line in open(data_path, 'r').readlines()]

    results = []
    # 每次取 batch_size 条数据
    for i, queries in enumerate([data[k: k + batch_size] for k in range(0, len(data), batch_size)]):
        queries = [q['text'] for q in queries]

        for j, question in enumerate(queries):
            print(f"Question-{i * batch_size + j + 1}: {question}")

        messages = [[
            {"role": "system", "content": '你是一个人工智能助手。'},
            {"role": "user", "content": q}
        ] for q in queries]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        )

        inputs = inputs.to(device)

        gen_kwargs = {"max_length": 512, "do_sample": True, "top_k": 1}

        with torch.no_grad():
            responses = model.generate(**inputs, **gen_kwargs)
            batch_results = []
            for response in responses:
                response = response[inputs['input_ids'].shape[1]:]
                response = tokenizer.decode(response, skip_special_tokens=True)
                response = response.strip()
                print(f'Response: {response}')
                batch_results.append(response)

        results.extend({
                           'id': len(results) + 1,
                           'question': question,
                           'response': response
                       } for question, response in zip(queries, batch_results))

    # 保存生成结果
    with open(out_file, 'w', encoding='utf-8') as f:  # 最好把 encoding='utf-8' 带上，否则可能报 UnicodeEncodeError 错误
        for line in results:
            print(line)
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    # 参数解析
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, default='data/seval.jsonl')
    argparser.add_argument('--batch_size', type=int, default=4)
    argparser.add_argument('--model_path', type=str, default='/mnt/data/djx/tfs_models/Qwen2-0.5B-Instruct')
    argparser.add_argument('--out_file', type=str, default='qwen_results.jsonl')
    args = argparser.parse_args()
    print(args)

    # 生成响应
    generate(
        model_path=args.model_path,
        data_path=args.data_path,
        batch_size=args.batch_size,
        out_file=args.out_file
    )
