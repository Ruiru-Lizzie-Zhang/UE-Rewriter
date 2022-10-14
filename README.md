# UE-Rewriter
Codes for paper UE-Rewriter

跑5个10%数据：
python generate_batch_wise.py --model_name "DialoGPT-small" --debug True

跑一个：
python generate_batch_wise.py --model_name "blenderbot_small-90M" --debug True


evaluate on bleu:
python eval.py --debug True
