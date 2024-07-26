import json
import argparse
import tqdm
import time
from openai import OpenAI
from os import path
import os
import json

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompt_path', type=str, default='prompt_template.txt')
    argparser.add_argument('--criteria_path', type=str, default='criteria.json')
    argparser.add_argument('--output_dir', type=str, default='eval_results')
    argparser.add_argument('--data_dir', type=str, default='data')
    argparser.add_argument('--eval-model', type=str, default='qwen2:72b')
    argparser.add_argument('--OPENAI_API_KEY', type=str, required=True)
    # argparser.add_argument('--eval_model', type=str, default='gpt-4')
    args = argparser.parse_args()

    data_path = path.join(args.data_dir, f'qa_{args.model}.jsonl')

    with open(data_path, 'r') as f:
        dataset = [json.loads(line) for line in f.readlines()]
    criteria = json.load(open(args.criteria_path))
    prompt_template = open(args.prompt_path).read()

    save_dir = args.save_dir
    if not path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = path.join(args.save_dir, f'res_{args.model}.jsonl')

    print(
        f'Data_path: {data_path} \n \
        Output_path: {save_path} \n \
        Model: {args.model} \n \
        Eval Model: {args.eval_model} \n '
    )

    # 创建 OpenAI 客户端
    client = OpenAI(
        api_key=args.OPENAI_API_KEY,
    )

    ignore = 0  # 统计无效评价结果的数量
    results = []  # 保存评价结果
    for i, instance in enumerate(dataset):
        print(f"{i+1}/{len(dataset)}")
        qa = f"问题：{instance['question']}\n回答：{instance['response']}"
        line = {}
        for k, v in criteria.items():
            prompt = prompt_template.replace('{{context}}', qa).replace('{{issue}}', k).replace('{{content}}', v)
            print(prompt)
            try:
                response = client.chat.completions.create(
                    model=args.eval_model,
                    messages=[
                        {'role': 'system', 'content': '你是一个问答质量评价助手。'},
                        {'role': 'user', 'content': prompt}
                    ]
                )
                response = response.choices[0].message.content
                print(response)

                line['criteria'] = k
                res1, res2 = response.split('解释：')
                score = res1.split('得分：')[-1].strip()
                explain = res2.strip()
                # print(score)
                # print(explain)
                line['response'] = {
                    'score': score,
                    'explain': explain
                }
            except Exception as e:
                print(e)
                if "limit" in str(e):
                    time.sleep(2)
                else:
                    ignore += 1
                    print('ignored', ignore)
                    break
        results.append(line)

    print('ignored total', ignore)
    with open(path.join(args.save_dir, f'{args.model}.jsonl'), 'w', encoding='utf-8') as f:
        for line in results:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
