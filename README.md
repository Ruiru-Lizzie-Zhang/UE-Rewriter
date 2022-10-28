# UE-Rewriter
## Codes for paper UE-Rewriter



---

Start with UE-rewriting data with BERT Masked Language Model. Only need data.json.zip OR data.json to be present.
```
python preprocess.py

python rewriter.py --data_dir part_cleaned_data
```


---

## For Lizzie

Under codes_punc/, run the following to generate rewritten_hypotheses_by_blenderbot_small-90M.txt:
```
python generate_batch_wise.py --data_dir unseen_from_bert-base-uncased_predicted_by_bert-base-uncased_rewritten_data.txt
```

Using another command line, run the following to generate original_hypotheses_by_blenderbot_small-90M.txt:
```
python generate_batch_wise.py --data_dir part_cleaned_data.txt --rewritten_ids_dir rewritten_ids.pt
```

Compare the evaluation of the two hypotheses with the SAME reference. Here reference are all sentences indexed by rewritten_ids.pt[+1].

---

## Irrelevance

Reminder: Modify the groundtruths in eval.py for rewritten data.

Reminder: top_k is testable.


Rewrite:
```
python rewriter.py --unseen_tokenizer_name 'bert-base-uncased' --pred_model_name 'bert-base-uncased' --data_dir ../data/all_data.txt --pos_dir ../data/pos.pt --rewrite_batch_size 128

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

---
新eval（bleuT在使用时要将文件名改回bleu）
···
python metric_evaluate.py -metric [指标名] -hyp [生成文件] -ref [ground truth]
···
指标名：chrF, rouge, meteor
···
