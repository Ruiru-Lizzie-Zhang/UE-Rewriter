

#read all data
docs=[]
import json
f = open('./data.json')
data = json.load(f)
  
for i in data:
    docs.append(i)

f.close()

#locate unseen entities
from tqdm import tqdm
import re
from transformers import BertTokenizer
import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
import pandas as pd

all_data = []
unseen_dataset = pd.DataFrame()
sentences=[]
unseen_entities=[]
doc_nums=[]
dialog_indices=[]

for doc_num in tqdm(range(len(data))):
    dialog = []
    for i in docs[doc_num]['dialog']:
        dialog.append(i['text'])
    dialog_lower = [text.lower() for text in dialog]
    all_data.append(dialog_lower)
    
    #build vocabulary
    text = ''.join(dialog)
    clean_text = re.sub(r"[,.;@#?!&$/]+\ *", " ", text)
    vocabulary = set(clean_text.lower().split())
    
    #BERT tokenization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    indexed_tokens = []
    for text in dialog:   
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens.append(tokenizer.convert_tokens_to_ids(tokenized_text))
    
    #BERT tokenized vocabulary
    compact = []
    for i in indexed_tokens:
        compact.extend(i) 
    tokenized_vocab = set(compact)
    
    #BERT text vocabulary
    new_vocab = set(tokenizer.convert_ids_to_tokens(tokenized_vocab))
    
    #unseen words in BERT
    unseen = vocabulary.difference(new_vocab)
    unseen = list(unseen)

    #find sentences with unseen entities
    for word in unseen:
        indices = [i for i, x in enumerate([word in i for i in dialog_lower]) if x == True] 
        for index in indices:
            sentence = dialog_lower[index]
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
import torch
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()
# model.to('cuda')  # if you have gpu

def predict_masked_sent(text, top_k=5):
    # Tokenize input
    text = "[CLS] %s [SEP]"%text
    tokenized_text = tokenizer.tokenize(text)
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
for dialog in all_data:
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
