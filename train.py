# !pip install transformers
# !pip install datasets
# !pip install evaluate
# !pip install rouge_score

from preprocess import read_txt, file_exist
import warnings
warnings.filterwarnings("ignore")
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from datasets import load_dataset
import evaluate
import numpy as np
np.random.seed(12345)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

import argparse
from argparse import RawTextHelpFormatter
def parse_option():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--data_dir_txt', type=str, default='../data/all_data_punc.txt')
    parser.add_argument('--data_dir_json', type=str, default='../data/dataset.json')
    parser.add_argument('--eod_token', type=str, default='##')
    parser.add_argument('--metric', type=str, default='rouge')
    parser.add_argument('--model_name', type=str, default="blenderbot_small-90M")
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=5)

    opt = parser.parse_args()
    return opt

opt = parse_option()
print(opt)

tokenizer = AutoTokenizer.from_pretrained("facebook/"+opt.model_name)
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

        if opt.metric == "rouge":
            result = {key: value * 100 for key, value in result.items()}
        else:
            result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        return result


def main():

    if not file_exist(opt.data_dir_json):
        from preprocess import txt_to_json_pair
        txt_to_json_pair(opt.data_dir_txt, opt.data_dir_json, opt.eod_token)
    
    dataset = load_dataset("json", data_files=opt.data_dir_json, split='train')
    dataset = dataset.train_test_split(test_size=0.2, seed=12345)
    tokenized_datasets = dataset.map(preprocess_function).remove_columns(['input','output'])

    metric = evaluate.load(opt.metric)
    
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/"+opt.model_name)
    model = torch.nn.DataParallel(model).to(DEVICE)
    
    training_args = Seq2SeqTrainingArguments(output_dir="trainer", 
                                             evaluation_strategy="epoch",
                                             optim="adamw_torch",
                                             num_train_epochs=opt.num_epochs,
                                             #logging_steps=50,
                                             per_device_train_batch_size=opt.train_batch_size,
                                             per_device_eval_batch_size=opt.eval_batch_size)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics,
    )

    print('--- begin training ---')
    #torch.cuda.empty_cache()
    train_result = trainer.train()
    trainer.save_model('model.pt')
    torch.save(train_result, 'results.pt')
    
    
if __name__ == '__main__':
    main()
    
