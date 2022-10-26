import re
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from tqdm import tqdm
from collections import Counter
from transformers import BertTokenizer, BertForMaskedLM
from preprocess import read_txt, file_exist
from operator import itemgetter

import torch
from torch.nn.functional import softmax
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import argparse
from argparse import RawTextHelpFormatter
def parse_option():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='all_data.txt')
    parser.add_argument('--pos_dir', type=str, default='pos.pt')
    parser.add_argument('--unseen_tokenizer_name', type=str, default='bert-base-uncased')
    parser.add_argument('--pred_model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--eod_token', type=str, default='##')
    parser.add_argument('--num_top_unseen', type=int, default=20)
    parser.add_argument('--idx_pred_mask', type=int, default=1)
    parser.add_argument('--entity_only', type=bool, default=True)
    parser.add_argument('--rewrite', type=bool, default=True)
    parser.add_argument('--mask_batch_size', type=int, default=128)
    parser.add_argument('--rewrite_batch_size', type=int, default=64)
    parser.add_argument('--demo', type=bool, default=False)
    parser.add_argument('--window_size', type=int, default=1)

    opt = parser.parse_args()
    return opt

def get_unseen_words(all_data, tokenizer):
    '''
    all_data: list of strings, lower-case, cleaned
    '''
    vocab = set(' '.join(all_data).split())
    input_ids = tokenizer(list(vocab), add_special_tokens=False, 
                          return_token_type_ids=False, return_attention_mask=False)["input_ids"]
    tokenized_vocab_id = set([token for tokenized_word in input_ids for token in tokenized_word])
    tokenized_vocab = tokenizer.convert_ids_to_tokens(tokenized_vocab_id)
    # eod token is also unseen for tokenizers
    return list(vocab.difference(tokenized_vocab))
        
#     indexed_tokens = tokenizer(all_data)['input_ids']
#     for batch_id in tqdm(range(0, len(all_data), opt.token_batch_size)):
#         batch = all_data[batch_id: batch_id + opt.token_batch_size]
#         input_ids = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")["input_ids"]
#         tokenized_vocab = set(batch)
#         new_vocab = set(tokenizer.convert_ids_to_tokens(tokenized_vocab))
        
#         special_tokens = tokenizer.special_tokens_map.values() # not needed, as special tokens are not in original vocab
#         new_vocab = new_vocab.difference(special_tokens)
    
def MLM_demo(text, top_k, tokenizer, model):
    # Tokenize input
    text = "[CLS] %s [SEP]"%text
    tokenized_text = tokenizer.tokenize(text)
    #print(tokenized_text)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    # tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

    pred_with_prob = {}
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        token_weight = top_k_weights[i]
        pred_with_prob[predicted_token] = float(token_weight)
        #print("[MASK]: '%s'"%predicted_token, " | weights:", float(token_weight))
    return pred_with_prob


def rewrite_batch(batch, top_k, tokenizer, model):
    
    input_ids = tokenizer(batch, padding='max_length', truncation=True, return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids)['logits']
        probs = softmax(outputs, dim=-1)
    
    mask_token_id = tokenizer.mask_token_id
    mask_indices = torch.nonzero(torch.eq(input_ids, mask_token_id))

    for idx_sen, idx_word in mask_indices:
        _, top_k_indices = torch.topk(probs[idx_sen, idx_word], top_k, sorted=True)
        input_ids[idx_sen, idx_word] = top_k_indices[top_k-1]

    return tokenizer.batch_decode(input_ids, skip_special_tokens=True)


def main():
    opt = parse_option()
    
    all_data = read_txt(opt.data_dir)
    all_words = ' '.join(all_data).split()
    print(f"Total words: {len(all_words)}")
    
    unseen_dir = opt.unseen_tokenizer_name+'_unseen_words.txt'
    if file_exist(unseen_dir):
        with open(unseen_dir, 'r') as f:
            unseen = f.read().split('\n')
            f.close()
    else:
        print('Unseen vocabulary not found. Building unseen vocabulary'+''.join(['-']*100))
        if 'bert' in opt.unseen_tokenizer_name.lower():
            tokenizer = BertTokenizer.from_pretrained(opt.unseen_tokenizer_name)
        elif 'blender' in opt.unseen_tokenizer_name.lower(): # eg. "blenderbot_small-90M"
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("facebook/"+opt.unseen_tokenizer_name)
        elif 'gpt' in opt.unseen_tokenizer_name.lower(): # eg. "DialoGPT-small"
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("microsoft/"+opt.unseen_tokenizer_name)
        else:
            raise ValueError('Unsupported tokenizer name to build unseen vocabulary.')
        unseen = get_unseen_words(all_data, tokenizer)
        with open(unseen_dir, 'w') as f:
            f.write('\n'.join(unseen))
            f.close()
    while opt.eod_token in unseen:
        unseen.remove(opt.eod_token)
    print(f"Total unseen words: {len(unseen)}")
    
    c = Counter(all_words)
    unseen_count = {w: c[w] for w in unseen}
    unseen_count = {w: count for w, count in sorted(unseen_count.items(), key=lambda item: item[1], reverse=True)}
    print(f"{opt.num_top_unseen} most frequent unseen words and counts: {list(unseen_count.items())[:opt.num_top_unseen]}")
        
    if opt.entity_only:
        pos = torch.load(opt.pos_dir)
        pos_flatten = [word_pos for sen_pos in pos for word_pos in sen_pos]
        entity_tokens = ['NN', 'NNS', 'NNP', 'NNPS']
        all_entities = [word for word, s in pos_flatten if s in entity_tokens and word != '##']
        print(f"Total entities: {len(all_entities)}")
        
        unseen = list(set(unseen).intersection(all_entities))
        print(f"Total unseen entities: {len(unseen)}")
        unseen_count = {w: c[w] for w in unseen}
        unseen_count = {w: count for w, count in sorted(unseen_count.items(), key=lambda item: item[1], reverse=True)}
        print(f"{opt.num_top_unseen} most frequent unseen entities and counts: {list(unseen_count.items())[:opt.num_top_unseen]}")
        
        
    if opt.rewrite:
#         unseen_dataset = pd.DataFrame()
#         sentences=[]
#         unseen_entities=[]
#         doc_nums=[]
#         dialog_indices=[]
#         #locate unseen entities
#         '''
#           doc_num: index of dialog in the dataset
#           dialog_num: index of sentence in a dialog
#         '''
#         unseen_dataset['unseen entity'] = unseen_entities
#         #unseen_dataset['sentence'] = sentences
#         unseen_dataset['doc number'] = doc_nums
#         unseen_dataset['dialog index'] = dialog_indices
#         unseen_dataset.to_csv("unseen_ids.csv", index=False)


        #Masked Language Model
        tokenizer = BertTokenizer.from_pretrained(opt.pred_model_name)
        mask_token = tokenizer.mask_token
        
        model = BertForMaskedLM.from_pretrained(opt.pred_model_name).to(DEVICE)
        model.eval()
        # model.to('cuda')  # if you have gpu

        #rewrite unseen entities
        #ex = [] #long sentence cannot be tokenized

#         for i in tqdm(range(len(unseen_dataset))):
#             doc_num = unseen_dataset['doc number'][i]
#             dialog_num = unseen_dataset['dialog index'][i]
#             unseen_entity = unseen_dataset['unseen entity'][i]

#             unseen_sentence = all_data[doc_num][dialog_num]

        #if opt.window_size == 0: # or dialog_num < window_size:
        
        if opt.demo:
            print('Masking'+''.join(['-']*100))
            for UE in tqdm(unseen):
                all_data = re.sub('\\b'+UE+'\\b', mask_token, '\n'.join(all_data)).split('\n')
                mask_sentence_indices = [i for i, j in enumerate(mask_data) if '[MASK]' in j]
                if len(mask_sentence_indices) == 1:
                    mask_sentences = [mask_data[mask_sentence_indices[0]]]
                else:
                    mask_sentences = list(itemgetter(*mask_sentence_indices)(mask_data))
                try:
                    UE_pred_list = [list(MLM_demo(s, opt.idx_pred_mask, 
                                                            tokenizer, model).keys())[-1] for s in mask_sentences]
                    for idx in mask_sentence_indices:
                        mask_data[idx] = mask_data[idx].replace(mask_token, UE_pred_list.pop(0)) 
                except RuntimeError:
                    for idx in mask_sentence_indices:
                        mask_data[idx] = mask_data[idx].replace(mask_token, UE_pred) 
                    print("--- Long sentence encountered; unseen entity kept.")
    #                 for idx in mask_sentence_indices:
    #                     #print(mask_data[idx])
    #                     try:
    #                         mask_sentence = mask_data[idx]
    #                         pred = predict_masked_sen(mask_sentence, top_k=opt.idx_pred_mask, tokenizer=tokenizer, model=model)
    #                         UE_pred = list(pred.keys())[-1]
    #                     except RuntimeError:
    #                         UE_pred = UE
    #                         print("--- Long sentence encountered; unseen entity kept.")
    #                    mask_data[idx] = mask_sentence.replace('[MASK]', UE_pred)
                    #print(mask_data[idx])
                all_data = mask_data
            #ex.append((doc_num, dialog_num))
            with open('unseen_from_'+opt.unseen_tokenizer_name+'_predicted_by_'+opt.pred_model_name+'_rewritten_data.txt', 'w') as f:
                f.write('\n'.join(all_data))
                f.close()

                
        else: # faster
            mask_dir = 'masked_all_data_by_'+unseen_dir
            if file_exist(mask_dir):
                with open(mask_dir, 'r') as f:
                    all_data_str = f.read()
                    f.close()
            else:
                print('Masking'+''.join(['-']*100))
                all_data_str = '\n'.join(all_data)
                for batch_id in tqdm(range(0, len(unseen), opt.mask_batch_size)):
                    unseen_batch = unseen[batch_id: batch_id + opt.mask_batch_size]
                    pattern = re.compile('|'.join(['\\b'+re.escape(w)+'\\b' for w in unseen_batch])) #when consider punctuations
                    #pattern = re.compile('|'.join(['\\b'+w+'\\b' for w in unseen_batch]))
                    all_data_str = pattern.sub(' '+mask_token+' ', all_data_str)
                    #print(all_data_str[:10])
                with open(mask_dir, 'w') as f:
                    f.write(all_data_str)
                    f.close()
                
            all_data = all_data_str.split('\n')
            if opt.window_size > 1:
                print('Adding context'+''.join(['-']*100)) 
                eod_id = [i for i, sen in enumerate(all_data) if sen == opt.eod_token]
                ids = []
                for w in range(opt.window_size):
                    ids.extend([idx-w for idx in eod_id if idx>=w])
                all_data = [' '.join(all_data[i:i+opt.window_size]) for i in tqdm(range(len(all_data))) if i not in ids]
                print('Saving indices for references'+''.join(['-']*100)) 
                ref_ids = [i+opt.window_size for i in tqdm(range(len(all_data))) if i not in ids]
                torch.save(ref_ids, 'ref_ids_window_'+str(opt.window_size)+'.pt')
            
            print('Rewriting'+''.join(['-']*100))                
            with open('unseen_from_'+opt.unseen_tokenizer_name+'_predicted_by_'+opt.pred_model_name+'_rewritten_data.txt', 'w') as f:
                f.write('')
            for batch_id in tqdm(range(0, len(all_data), opt.rewrite_batch_size)):
                batch = all_data[batch_id: batch_id + opt.rewrite_batch_size]
                rewritten_batch = rewrite_batch(batch, opt.idx_pred_mask, tokenizer, model)
                with open('unseen_from_'+opt.unseen_tokenizer_name+'_predicted_by_'+opt.pred_model_name+'_rewritten_data.txt', 'a') as f:
                    f.write('\n'.join(rewritten_batch))
                    f.write('\n')

#         else:
#             pass
#                 context = ""
#                 for sen in all_data[doc_num][dialog_num-window_size : dialog_num]:
#                     context = context+sen
#                 mask_sentence = context+mask_sentence
#                 try:
#                     pred = predict_masked_sen(mask_sentence, top_k=opt.idx_pred_mask)
#                     UE_pred = list(pred.keys())[0]
#                 except RuntimeError:
#                     UE_pred = unseen_entity
#                     ex.append((doc_num, dialog_num))

#             rewrited_sentence = mask_sentence.replace('[MASK]', UE_pred)
#             del all_data[doc_num][dialog_num]
#             all_data[doc_num].insert(dialog_num, rewrited_sentence)

        #save rewritten data
#         file = open('./rewrited_data_w0.txt','w')
#         for dialog in all_data: # all_data is a list of dialogs (each dialog is a list of sentences)
#             for sen in dialog:
#                 file.write(sen)
#                 file.write('\n')
#             file.write('##')
#             file.write('\n')
#         file.close()

#         #ensure size of data remains the same
#             #read rewrited data
#         file = open('./rewrited_data_w0.txt','r')
#         file_data = file.read() 
#         file_data = file_data.split('##')

#         while "" in file_data:
#             file_data.remove("")

#         data = []
#         first = file_data[0].split('\n')
#         del(first[-1])
#         data.append(first)

#         for dialog in file_data[1:]:
#             tep_list = dialog.split('\n')
#             del(tep_list[0])
#             del(tep_list[-1])
#             data.append(tep_list)

#         del(data[-1])


#         assert [len(i) for i in all_data] == [len(i) for i in data]

    
if __name__ == '__main__':
    main()
