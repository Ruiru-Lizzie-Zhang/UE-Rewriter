import re
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from tqdm import tqdm

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import argparse
from argparse import RawTextHelpFormatter
def parse_option():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='all_data.txt')
    #parser.add_argument('--unseen_dir', type=str, default=None)
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased')
    #parser.add_argument('--token_batch_size', type=int, default=64)
    parser.add_argument('--rewrite', type=bool, default=True)
    parser.add_argument('--debug', type=bool, default=False)

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
    
def predict_masked_sent(text, tokenizer, top_k=5):
    # Tokenize input
    text = "[CLS] %s [SEP]"%text
    tokenized_text = tokenizer.tokenize(text)
    print(tokenized_text)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    # tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

    pred_with_prob = {}
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        token_weight = top_k_weights[i]
        pred_with_prob[predicted_token] = float(token_weight)
        #print("[MASK]: '%s'"%predicted_token, " | weights:", float(token_weight))
    return pred_with_prob

def main():
    opt = parse_option()
    
    from preprocess import read_txt, file_exist
    all_data = read_txt(opt.data_dir)
    
    if 'bert' in opt.tokenizer_name.lower():
        from transformers import BertTokenizer, BertForMaskedLM
        tokenizer = BertTokenizer.from_pretrained(opt.tokenizer_name)
    elif 'blender' in opt.tokenizer_name.lower(): # eg. "blenderbot_small-90M"
        from transformers import AutoTokenizer, BertForMaskedLM
        tokenizer = AutoTokenizer.from_pretrained("facebook/"+opt.tokenizer_name)
    elif 'gpt' in opt.tokenizer_name.lower(): # eg. "DialoGPT-small"
        from transformers import AutoTokenizer, BertForMaskedLM
        tokenizer = AutoTokenizer.from_pretrained("microsoft/"+opt.tokenizer_name)
    else:
        raise ValueError('Unsupported tokenizer name')
        
    unseen_dir = opt.tokenizer_name+'_unseen_words.txt'
    if file_exist(unseen_dir):
        with open(unseen_dir, 'r') as f:
            unseen = f.read().split('\n')
            f.close()
    else:
        unseen = get_unseen_words(all_data, tokenizer)
        with open(unseen_dir, 'w') as f:
            f.write('\n'.join(unseen))
            f.close()
        
    if opt.rewrite:
        print(str(opt.rewrite))
        unseen_dataset = pd.DataFrame()
        sentences=[]
        unseen_entities=[]
        doc_nums=[]
        dialog_indices=[]
        #locate unseen entities
        '''
          doc_num: index of dialog in the dataset
          dialog_num: index of sentence in a dialog
        '''
        for doc_num, dialog in tqdm(enumerate(all_data)):
            #find sentences with unseen entities
            for word in unseen:
                #indices = [i for i, x in enumerate([word in i for i in dialog]) if x == True] 
                indices = [i for i, sen in enumerate(dialog) if word in sen]
                for index in indices:
                    sentence = dialog[index]
                    result = nltk.pos_tag(nltk.word_tokenize(sentence))
                    result = dict(result)
                    if word in result:
                        if result[word] in ['NN', 'NNS', 'NNP', 'NNPS']:
                            sentences.append(dialog[index])
                            unseen_entities.append(word)
                            doc_nums.append(doc_num)
                            dialog_indices.append(index)

        unseen_dataset['unseen entity'] = unseen_entities
        #unseen_dataset['sentence'] = sentences
        unseen_dataset['doc number'] = doc_nums
        unseen_dataset['dialog index'] = dialog_indices
        unseen_dataset.to_csv("unseen_ids.csv", index=False)


        #Masked Language Model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        model.eval()
        # model.to('cuda')  # if you have gpu

        #rewrite unseen entities
        window_size = 0
        ex = [] #long sentence cannot be tokenized

        for i in tqdm(range(len(unseen_dataset))):
            doc_num = unseen_dataset['doc number'][i]
            dialog_num = unseen_dataset['dialog index'][i]
            unseen_entity = unseen_dataset['unseen entity'][i]

            unseen_sentence = all_data[doc_num][dialog_num]
            mask_sentence = re.sub('\\b'+unseen_entity+'\\b', '[MASK]', unseen_sentence)

            if window_size==0 or dialog_num < window_size:
                try:
                    pred = predict_masked_sent(mask_sentence, top_k=1)
                    UE_pred = list(pred.keys())[0]
                except RuntimeError:
                    UE_pred = unseen_entity
                    ex.append((doc_num, dialog_num))

            else:
                context = ""
                for sen in all_data[doc_num][dialog_num-window_size : dialog_num]:
                    context = context+sen
                mask_sentence = context+mask_sentence
                try:
                    pred = predict_masked_sent(mask_sentence, top_k=1)
                    UE_pred = list(pred.keys())[0]
                except RuntimeError:
                    UE_pred = unseen_entity
                    ex.append((doc_num, dialog_num))

            rewrited_sentence = mask_sentence.replace('[MASK]', UE_pred)
            del all_data[doc_num][dialog_num]
            all_data[doc_num].insert(dialog_num, rewrited_sentence)

        #save rewritten data
        file = open('./rewrited_data_w0.txt','w')
        for dialog in all_data: # all_data is a list of dialogs (each dialog is a list of sentences)
            for sen in dialog:
                file.write(sen)
                file.write('\n')
            file.write('##')
            file.write('\n')
        file.close()

        #ensure size of data remains the same
            #read rewrited data
        file = open('./rewrited_data_w0.txt','r')
        file_data = file.read() 
        file_data = file_data.split('##')

        while "" in file_data:
            file_data.remove("")

        data = []
        first = file_data[0].split('\n')
        del(first[-1])
        data.append(first)

        for dialog in file_data[1:]:
            tep_list = dialog.split('\n')
            del(tep_list[0])
            del(tep_list[-1])
            data.append(tep_list)

        del(data[-1])


        assert [len(i) for i in all_data] == [len(i) for i in data]

    
if __name__ == '__main__':
    main()
