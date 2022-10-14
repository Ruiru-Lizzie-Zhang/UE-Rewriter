# UE-Rewriter
Codes for paper UE-Rewriter

跑5个10%数据：
```
python generate_batch_wise.py --model_name "DialoGPT-small" --debug True
```

跑一个：
```
python generate_batch_wise.py --model_name "blenderbot_small-90M" --debug True
```

evaluate on bleu:
```
python eval.py --debug True
python eval.py --hyp_dir 'blenderbot_small-90M_generate_rewrited.txt' --ref_dir 包含##的rewrited_data --debug True
```
