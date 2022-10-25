# UE-Rewriter
Codes for paper UE-Rewriter

Reminder: Modify the groundtruths in eval.py for rewritten data.

Reminder: top_k is testable.

---

Rewrite:
```
python rewriter.py --unseen_tokenizer_name 'bert-base-uncased' --pred_model_name 'bert-base-uncased' --data_dir data/all_data.txt --pos_dir data/pos.pt --rewrite_batch_size 128

python rewriter.py --unseen_tokenizer_name 'blenderbot_small-90M' --pred_model_name 'bert-base-uncased'

python rewriter.py --unseen_tokenizer_name 'DialoGPT-small' --pred_model_name 'bert-base-uncased'
```

---

跑5个10%数据：
```
python generate_batch_wise.py --model_name "DialoGPT-small" --debug True
```

Modify the following to generate an output txt for a given input txt.
```
python generate_batch_wise.py --model_name "blenderbot_small-90M" --data_dir 'all_data.txt'
```

evaluate original data via bleu:
```
python eval.py --debug True
```
evaluate rewrited data via bleu:
```
python eval.py --hyp_dir 'blenderbot_small-90M_generate_rewrited.txt' --ref_dir 包含##的rewrited_data --debug True
```
