from tqdm import tqdm


def file_exist(directory):
    import os
    if os.path.isfile(directory):
        return True
    return False


def read_json(directory, clean, to_txt=False, eod_token='##'):
    
    import json
    # read data as list of list, each smaller list is a dialog composing of X sentences (8<=X<=23 for Wizard of Wikipedia)
    
    f = open(directory)
    docs = list(json.load(f))
    f.close()

    all_data = []
    if clean.lower() == 'all':
        import re
        for doc in tqdm(docs):
            dialog = [re.sub(r"\W", " ", i['text']).lower() for i in doc['dialog']]
            all_data.append(dialog)
    elif clean.lower() == 'part':
        import re
        for doc in tqdm(docs):
            dialog = [re.sub(r"([!,.:;?])", r' \1 ', re.sub(r"[^\w!,.:;?]", ' ', i['text'])).lower() for i in doc['dialog']] # keep !,.:;?
            all_data.append(dialog)
    else:
        for doc in tqdm(docs):
            dialog = [i['text'].lower() for i in doc['dialog']]
            all_data.append(dialog)  
    
    if to_txt:
        save_dir = clean.lower()+'_cleaned_data.txt'
        with open(save_dir, 'w') as f:
            f.write('')
        with open(save_dir, 'a') as f:
            for dialog in all_data:
                f.write('\n'.join(dialog))
                f.write('\n'+eod_token+'\n')

    return all_data


def read_txt(directory):
    
    # read data as list of CLEANED LOWER-CASE sentences, separated by ##
    
    file = open(directory, 'r', encoding='utf-8')
    file_data = file.read() 
    all_data = file_data.split('\n')

    while "" in all_data:
        all_data.remove("")
    return all_data


def get_pos(data, save_dir='pos.pt'):
    
    import torch
    import nltk
    try:
        pos = [nltk.pos_tag(sen.split()) for sen in tqdm(data)]
    except LookupError:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        pos = [nltk.pos_tag(sen.split()) for sen in tqdm(data)]
    
    torch.save(pos, save_dir)
    return pos


if __name__ == '__main__':
    all_data = read_json('data.json', clean='part', to_txt=True)
