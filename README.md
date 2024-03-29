# UE-Rewriter
## Codes for paper UE-Rewriter



---

Start with UE-rewriting data with BERT Masked Language Model. Only need `data.json.zip` OR `data.json` to be present.
```
python preprocess.py
```

---

Rewrite:
```
python rewriter.py --data_dir part_cleaned_data
```
<!-- 
python rewriter.py --unseen_tokenizer_name 'bert-base-uncased' --pred_model_name 'bert-base-uncased' --data_dir ../data/all_data.txt --pos_dir ../data/pos.pt --rewrite_batch_size 128

python rewriter.py --unseen_tokenizer_name 'blenderbot_small-90M' --pred_model_name 'bert-base-uncased'

python rewriter.py --unseen_tokenizer_name 'DialoGPT-small' --pred_model_name 'bert-base-uncased' -->

---

To generate hypotheses on <b>rewritten</b> inputs using various benchamark models:
```
python generate_batch_wise.py --data_dir unseen_from_bert-base-uncased_predicted_by_bert-base-uncased_rewritten_data.txt --model_name blenderbot_small-90M

python generate_batch_wise.py --data_dir unseen_from_bert-base-uncased_predicted_by_bert-base-uncased_rewritten_data.txt --model_name blenderbot-400M-distill

python generate_batch_wise.py --data_dir unseen_from_bert-base-uncased_predicted_by_bert-base-uncased_rewritten_data.txt --model_name blenderbot-1B-distill --eval_batch_size 32

python generate_batch_wise.py --data_dir unseen_from_bert-base-uncased_predicted_by_bert-base-uncased_rewritten_data.txt --rewritten_ids_dir rewritten_ids.pt --model_name DialoGPT-small

python generate_batch_wise.py --data_dir unseen_from_bert-base-uncased_predicted_by_bert-base-uncased_rewritten_data.txt --rewritten_ids_dir rewritten_ids.pt --model_name DialoGPT-medium

python generate_batch_wise.py --data_dir unseen_from_bert-base-uncased_predicted_by_bert-base-uncased_rewritten_data.txt --rewritten_ids_dir rewritten_ids.pt --model_name DialoGPT-large
```

To generate hypotheses on <b>original</b> inputs using various benchamark models:
```
python generate_batch_wise.py --data_dir part_cleaned_data.txt --rewritten_ids_dir rewritten_ids.pt --model_name blenderbot_small-90M

python generate_batch_wise.py --data_dir part_cleaned_data.txt --rewritten_ids_dir rewritten_ids.pt --model_name blenderbot-400M-distill

python generate_batch_wise.py --data_dir part_cleaned_data.txt --rewritten_ids_dir rewritten_ids.pt --model_name blenderbot-1B-distill --eval_batch_size 32

python generate_batch_wise.py --data_dir part_cleaned_data.txt --rewritten_ids_dir rewritten_ids.pt --model_name DialoGPT-small

python generate_batch_wise.py --data_dir part_cleaned_data.txt --rewritten_ids_dir rewritten_ids.pt --model_name DialoGPT-medium

python generate_batch_wise.py --data_dir part_cleaned_data.txt --rewritten_ids_dir rewritten_ids.pt --model_name DialoGPT-large
```

---
<!-- 
## For Lizzie

1. (~1h) Under codes_punc/, run the following to generate rewritten_hypotheses_by_blenderbot_small-90M.txt:
```
python generate_batch_wise.py --data_dir unseen_from_bert-base-uncased_predicted_by_bert-base-uncased_rewritten_data.txt
```

2. (~1h simultaneously) Using another command line in the same directory codes_punc/, run the following to generate original_hypotheses_by_blenderbot_small-90M.txt:
```
python generate_batch_wise.py --data_dir part_cleaned_data.txt --rewritten_ids_dir rewritten_ids.pt
```

3. Compare the evaluation of the two hypotheses with the SAME reference. Here you need to generate a txt file for reference, which include all sentences in part_cleaned_data.txt with index i+1 for i in torch.load('rewritten_ids.pt').
Use both eval.py and metric_evaluate.py. The results should be the same (otherwise there are bugs).

4. Repeat 1-3 with --model_name 'DialoGPT-small'

---

## Irrelevance

Reminder: Modify the groundtruths in eval.py for rewritten data.

Reminder: top_k is testable. -->



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


---
## Fine Tuning

Original:
```
python train.py
```
Rewritten:
```
python train.py --data_dir_txt ../data/all_data_punc_rewritten.txt --eod_token '# #'
```

---
## Generate using fine-tuned model

Original:
```
python generate_batch_wise.py --data_dir all_data_punc.txt --model_ckpt pytorch_model.bin
```
Rewritten (using another checkpoint `pytorch_model.bin`):
```
python generate_batch_wise.py --data_dir all_data_punc_rewritten.txt --model_ckpt pytorch_model.bin
```
