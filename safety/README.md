# Safety evaluation based on GPT-4
A concise script prompting GPT-4 to evaluate the large language models' safety.

## Generation
```
python generate.py \
  --data_path prompt_template.txt \
  --model_path <model_path> \
  --output_path <output_path>
```

## Evaluation
```
python eval.py \
  --prompt_path prompt_template.txt \
  --criteria_path criteria.json \
  --OPENAI_API_KEY <your openai-api key> \
  --eval_model gpt-4
```

