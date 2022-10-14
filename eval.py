import os
import codecs
from operator import itemgetter
from bleu import corpus_bleu

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

def file_exist(pf):
    if os.path.isfile(pf):
        return True
    return False

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
    
    ref_data = list(map(list, zip(ref_data)))

    bleu, addition = corpus_bleu(hyp_data, ref_data)
    print("BLEU = %.2f, BLEU_1 = %.1f, BLEU_2 =%.1f, BLEU_3 = %.1f, BLEU_4 = %.1f \
    (BP=%.3f, ratio=%.3f, hyp_len=%d, ref_len=%d)"%(bleu[0]*100, bleu[1]*100, bleu[2]*100, bleu[3]*100, bleu[4]*100, 
                                                    addition[0], addition[1], addition[2], addition[3]))
    
    
    
if __name__ == '__main__':
    main()
