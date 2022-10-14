from preprocess import read_data
import torch
import argparse
from argparse import RawTextHelpFormatter

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # AutoModelForCausalLM
from tqdm import tqdm



def parse_option():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='../wizard_of_wikipedia/data.json')
    parser.add_argument('--model_name', type=str, default="facebook/blenderbot_small-90M")
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--min_len_generated', type=int, default=10)
    parser.add_argument('--max_len_generated', type=int, default=50)

    opt = parser.parse_args()
    return opt

def main():
    opt = parse_option()

    tokenizer = AutoTokenizer.from_pretrained(opt.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(opt.model_name)

    all_outputs = []
    for inputs in tqdm(read_data(opt.data_dir)):
        inputs = inputs[:-1]
        assert type(inputs) == list
        input_ids = tokenizer(inputs, padding='max_length', truncation=True, return_tensors="pt")["input_ids"]
        outputs = model.generate(input_ids=input_ids, num_beams=opt.num_beams, num_return_sequences=opt.num_return_sequences, 
                                 min_length=opt.min_len_generated, max_length=opt.max_len_generated)
        print(f"Outputs have shape {tuple(outputs.shape)}")
        all_outputs.append(outputs)
        
    torch.save(all_outputs, opt.model_name+'/output_ids.pt')

    
if __name__ == '__main__':
    main()
