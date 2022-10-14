from preprocess import read_txt
from tqdm import tqdm
#from bleu import corpus_bleu
import warnings
warnings.filterwarnings("ignore")
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import argparse
from argparse import RawTextHelpFormatter
def parse_option():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='all_data.txt')
    parser.add_argument('--model_name', type=str, default="DialoGPT-small")
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--min_len_generated', type=int, default=10)
    parser.add_argument('--max_len_generated', type=int, default=50)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--debug', type=bool, default=False)

    opt = parser.parse_args()
    return opt

def main():
    opt = parse_option()
    
    if opt.debug:
        all_data = read_txt(opt.data_dir)[:2000] # needs to be txt input
    else:
        all_data = read_txt(opt.data_dir)

    if 'blender' in opt.model_name.lower(): # eg. "blenderbot_small-90M"
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained("facebook/"+opt.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/"+opt.model_name).to(DEVICE)
    elif 'gpt' in opt.model_name.lower(): # eg. "DialoGPT-small"
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("microsoft/"+opt.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("microsoft/"+opt.model_name).to(DEVICE)
    else:
        raise ValueError('Unsupported model name')

#     all_outputs = []
#     bleus = []
    with open(opt.model_name+'_generate.txt', 'w') as f:
        f.write('')
    
    if 'blender' in opt.model_name.lower():
        for batch_id in tqdm(range(0, len(all_data), opt.eval_batch_size)):
            batch = all_data[batch_id: batch_id + opt.eval_batch_size]

            #input_ids = tokenizer(batch[:-1], padding='max_length', truncation=True, return_tensors="pt")["input_ids"]
            input_ids = tokenizer(batch, padding='max_length', truncation=True, return_tensors="pt")["input_ids"]
            input_ids = input_ids.to(DEVICE)
            output_ids = model.generate(input_ids=input_ids, num_beams=opt.num_beams, num_return_sequences=opt.num_return_sequences, 
                                        min_length=opt.min_len_generated, max_length=opt.max_len_generated)
            #all_outputs.append(output_ids)
            hyp = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            #ref = batch[1:]
    #         bleu = corpus_bleu(hyp, ref)[0]
    #         if sum([0 != i for i in bleu]) == 1:
    #             if bleu[1] > 0:
    #                 bleus.append(bleu[1])
    #             else:
    #                 bleus.append(bleu)
    #         else:
    #             bleus.append(bleu)

            with open(opt.model_name+'_generate.txt', 'a') as f:
                f.write('\n'.join(hyp))
                f.write('\n')

    #         if batch_id % (100 * opt.eval_batch_size) == 0:
    #             print('We are now at Sentence ###'+str(batch_id))
    #             torch.save(all_outputs, opt.model_name+'_output_ids.pt')
    #             torch.save(bleus, opt.model_name+'_bleus.pt')
    
    elif 'gpt' in opt.model_name.lower(): # eg. "DialoGPT-small":
        # GPT has no padding token!
        for batch in tqdm(all_data):
            batch = batch + tokenizer.eos_token
            input_ids = tokenizer(batch, return_tensors="pt")['input_ids']
            input_ids = input_ids.to(DEVICE)
            output_ids = model.generate(input_ids=input_ids, num_beams=opt.num_beams, num_return_sequences=opt.num_return_sequences, 
                                        min_length=input_ids.shape[-1]+opt.min_len_generated, max_length=input_ids.shape[-1]+opt.max_len_generated,
                                        pad_token_id=tokenizer.eos_token_id)
            hyp = tokenizer.batch_decode(output_ids[:,input_ids.shape[-1]:], skip_special_tokens=True)

            with open(opt.model_name+'_generate.txt', 'a') as f:
                f.write('\n'.join(hyp))
                f.write('\n')

    
if __name__ == '__main__':
    main()
