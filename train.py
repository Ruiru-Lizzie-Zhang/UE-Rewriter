# !pip install transformers
# !pip install evaluate
# !pip install rouge_score

from preprocess import read_txt
import json

data_dir = '../data/all_data_punc.txt'
eod_token = '##'
all_data = read_txt(data_dir)#[:103]

all_data_pair = [{'input': s, 'output':all_data[i+1]}\
            for i, s in enumerate(all_data[:-1]) if all_data[i+1]!=eod_token and s!=eod_token]
with open("../data/dataset.json", "w") as f:
    json.dump(all_data_pair, f)
    f.close()
    
from datasets import load_dataset
import evaluate
dataset = load_dataset("json", data_files="../data/dataset.json", split='train')
dataset = dataset.train_test_split(test_size=0.2, seed=12345)

metric_name = "rouge" 
metric = evaluate.load(metric_name)

import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_name = 'blenderbot_small-90M'
tokenizer = AutoTokenizer.from_pretrained("facebook/"+model_name)

def preprocess_function(examples):
    inputs = examples['input']
    outputs = examples['output']
    model_inputs = tokenizer(text=inputs, padding='max_length', truncation=True)
    labels = tokenizer(text_target=outputs, padding='max_length', truncation=True)
#     if ignore_markers:
#         labels["input_ids"] = [
#             [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
#         ]
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function).remove_columns(['input','output'])

def compute_metrics(eval_preds):
    with torch.no_grad():
        logits, labels = eval_preds
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    #     if ignore_markers:
    #         labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)

        if metric_name == "rouge":
            result = {key: value * 100 for key, value in result.items()}
        else:
            result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        return result

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/"+model_name).to(DEVICE)
training_args = Seq2SeqTrainingArguments(output_dir="test_trainer", 
                                         evaluation_strategy="epoch",
                                         optim="adamw_torch",
                                         num_train_epochs=10,
                                         logging_steps=50)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
)

print('--- begin training ---')
torch.cuda.empty_cache()
train_result = trainer.train()
trainer.save_model('model.pt')
torch.save(train_result, 'results.pt')
