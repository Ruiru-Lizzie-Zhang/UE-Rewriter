import codecs
from operator import itemgetter
import numpy as np
from preprocess import file_exist

# evaluation metrics
from bleu import corpus_bleu
import evaluate
from rouge_score import rouge_scorer


import argparse
from argparse import RawTextHelpFormatter
def parse_option():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--hyp_dir', type=str, default='blenderbot_small-90M_generate_original.txt')
    parser.add_argument('--ref_dir', type=str, default='all_data.txt')
    parser.add_argument('--eod_token', type=str, default='##')
    parser.add_argument('--ignore_eod', type=bool, default=True)
    parser.add_argument('--debug', type=bool, default=False)

    opt = parser.parse_args()
    return opt


def main():
    opt = parse_option()
    
    if opt.hyp_dir == None:
        raise Exception("Missing hypotheses (generated).")
    if opt.ref_dir == None:
        raise Exception("Missing references (groundtruths).")
    
    data = []
    for dir in [opt.hyp_dir, opt.ref_dir]:
        if not file_exist(dir):
            raise Exception("File Not Found: %s"%(dir))
        f = codecs.open(dir, encoding="utf-8")
        if opt.debug:
            data.append(f.read().split('\n')[:200]) # do not use readlines, o.w. \n will be included
        else:
            data.append(f.read().split('\n'))
        f.close()
        
    hyp_data, ref_data = data
    if len(hyp_data) != len(ref_data):
        raise Exception(f"Hypotheses length {len(hyp_data)} does not match references length {len(ref_data)}.")
    
    if opt.ignore_eod:
        mask_ref = [i for i, j in enumerate(ref_data) if not (j == opt.eod_token or ref_data[i-1] == opt.eod_token or i == 0)]
        mask_hyp = [i for i, j in enumerate(ref_data) if not (j == opt.eod_token or i == len(ref_data)-1 or ref_data[i+1] == opt.eod_token)]
        if len(mask_hyp) != len(mask_ref):
            raise Exception(f"Check whether each dialog is ended by {opt.eod_token}.")
        hyp_data = list(itemgetter(*mask_hyp)(hyp_data))
        ref_data = list(itemgetter(*mask_ref)(ref_data))
    
    
    # rouge
    rouge_list = ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(rouge_list, use_stemmer=True)
    scores = [scorer.score(i, j) for i, j in zip(hyp_data, ref_data)]
    for rouge in rouge_list:
        precision, recall, fmeasure = np.mean([score[rouge] for score in scores], axis=0)
        print(f"{rouge} score: precision = {precision:.4f}, recall = {recall:.4f}, fmeasure = {fmeasure:.4f}")


    ref_data = list(map(list, zip(ref_data)))
    # BLEU
    bleu, addition = corpus_bleu(hyp_data, ref_data)
    print(f"BLEU score: {bleu[0]*100:.4f}\n\
BLEU1 score: {bleu[1]*100:.4f}\n\
BLEU2 score: {bleu[2]*100:.4f}\n\
BLEU3 score: {bleu[3]*100:.4f}\n\
BLEU4 score: {bleu[4]*100:.4f}\n\
BLEU BP: {addition[0]:.4f}\n\
BLEU ratio: {addition[1]:.4f}")

    # chrF
    chrf = evaluate.load("chrf")
    scores = chrf.compute(predictions=hyp_data, references=ref_data, word_order=0) #chrF
    scores_plus = chrf.compute(predictions=hyp_data, references=ref_data, word_order=2) #chrF++
    print(f"chrF score: {scores['score']:.4f}\n\
chrF++ score: {scores_plus['score']:.4f}")
    
    
if __name__ == '__main__':
    main()
