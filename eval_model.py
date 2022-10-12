from preprocess import read_txt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # AutoModelForCausalLM
from tqdm import tqdm
from bleu import corpus_bleu

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import argparse
from argparse import RawTextHelpFormatter
def parse_option():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='all_data.txt')
    parser.add_argument('--model_name', type=str, default="blenderbot_small-90M")
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--min_len_generated', type=int, default=10)
    parser.add_argument('--max_len_generated', type=int, default=50)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--debug', type=bool, default=False)

    opt = parser.parse_args()
    return opt

def main():
    opt = parse_option()
    
    if opt.debug:
        all_data = read_txt(opt.data_dir)[:2000] # needs to be txt input
    else:
        all_data = read_txt(opt.data_dir)

    tokenizer = AutoTokenizer.from_pretrained("facebook/"+opt.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/"+opt.model_name).to(DEVICE)

    all_outputs = []
    bleus = []
    
    for batch_id in tqdm(range(0, len(all_data), opt.eval_batch_size)):
        batch = all_data[batch_id: batch_id + opt.eval_batch_size]
        
        input_ids = tokenizer(batch[:-1], padding='max_length', truncation=True, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(DEVICE)
        output_ids = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=1, min_length=10, max_length=50)
        all_outputs.append(output_ids)

        hyp = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        ref = batch[1:]
        bleu = corpus_bleu(hyp, ref)[0]
        if sum([0 != i for i in bleu]) == 1:
            if bleu[1] > 0:
                bleus.append(bleu[1])
            else:
                bleus.append(bleu)
        else:
            bleus.append(bleu)

        if batch_id % (10 * opt.eval_batch_size) == 0:
            print('We are now at Sentence ###'+batch_id)
            torch.save(all_outputs, opt.model_name+'_output_ids.pt')
            torch.save(bleus, opt.model_name+'_bleus.pt')

    
if __name__ == '__main__':
    main()
