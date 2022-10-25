import evaluate
import os
import ipdb
from rouge_score import rouge_scorer
from numpy import mean
import argparse
from bleu import corpus_bleu
from bleu import *
import codecs
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def recall(pds, gds):
    metric = evaluate.load("recall")
    macro_recall = metric.compute(predictions=pds, references=gds, average='macro')
    micro_recall = metric.compute(predictions=pds, references=gds, average='micro')
    average_recall = metric.compute(predictions=pds, references=gds, average='samples')
    print("macro_recall score: %.4f; micro_recall score: %.4f; recall score: %.4f "%(macro_recall['recall'], micro_recall['recall'], average_recall['recall']))
    
def chrF(pds, gds):
    chrf = evaluate.load("chrf")
    scores =chrf.compute(predictions=pds, references=gds, word_order = 0) #chrF
    scores_plus =chrf.compute(predictions=pds, references=gds, word_order = 2)  #chrF++
    print("chrF score: %.4f; chrF++ score: %.4f "%(scores['score'], scores_plus['score']))
def nltkbleu(pds, gds):
    chencherry = SmoothingFunction()
    score = sentence_bleu(references=gds, hypothesis=pds, 
                                   smoothing_function=chencherry.method5)

    print("nltkbleu score: ",score)        

def rouge(pds, gds):
    scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    length = len(gds)
    for i in range(length):
        score = scorer.score(pds[i],gds[i])
        scores.append(score)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True) #rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'])
    # a dictionary that will contain the results
    results1 = {'precision': [], 'recall': [], 'fmeasure': []}
    results2 = {'precision': [], 'recall': [], 'fmeasure': []}
    results3 = {'precision': [], 'recall': [], 'fmeasure': []}

    # for each of the hypothesis and reference documents pair
    for (h, r) in zip(pds, gds):
        # computing the ROUGE
        score = scorer.score(h, r)
        # separating the measurements
        precision1, recall1, fmeasure1 = score['rouge1']
        # add them to the proper list in the dictionary
        results1['precision'].append(precision1)
        results1['recall'].append(recall1)
        results1['fmeasure'].append(fmeasure1)

        precision2, recall2, fmeasure2 = score['rouge2']
        results2['precision'].append(precision2)
        results2['recall'].append(recall2)
        results2['fmeasure'].append(fmeasure2)

        precision3, recall3, fmeasure3 = score['rougeL']
        results3['precision'].append(precision3)
        results3['recall'].append(recall3)
        results3['fmeasure'].append(fmeasure3)
    print("rouge-1: precision:%.4f, recall:%.4f, fmeasure:%.4f "%(mean(results1['precision']), mean(results1['recall']), mean(results1['fmeasure'])))
    print("rouge-2: precision:%.4f, recall:%.4f, fmeasure:%.4f "%(mean(results2['precision']), mean(results2['recall']), mean(results2['fmeasure'])))
    print("rouge-L: precision:%.4f, recall:%.4f, fmeasure:%.4f "%(mean(results3['precision']), mean(results3['recall']), mean(results3['fmeasure'])))

def rougebox(pds, gds):
    scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    length = len(gds)
    for i in range(length):
        score = scorer.score(pds[i],gds[i])
        scores.append(score)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True) #rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'])
    # a dictionary that will contain the results
    results= list()
    for (h, r) in zip(pds, gds):
        # computing the ROUGE
        score = scorer.score(h, r)
        precision, recall, fmeasure = score['rougeL']
        results.append(round(precision,4))
        # results3['recall'].append(recall3)
        # results3['fmeasure'].append(fmeasure3)
    # ipdb.set_trace()
    for i in range(len(pds)):
        if results[i]==1.0:
            print(i+1)
    print(results)


    # print("rouge-L: precision:%.4f, recall:%.4f, fmeasure:%.4f "%(mean(results3['precision']), mean(results3['recall']), mean(results3['fmeasure'])))


def googlebleu(pds, gds):
    google_bleu = evaluate.load("google_bleu")
    # length = len(gds)
    # for i in range(length):
    #     score = chrf.compute(predictions=[pds[i]], references=[gds[i]])
    #     #scorer.score(pds[i],gds[i])
    #     scores.append(score)
    scores =google_bleu.compute(predictions=pds, references=gds, min_len=4) #google bleu
    print("google bleu score: %.4f; "%(scores['google_bleu']))

def bleu(pds, gds):
    bleu, addition = corpus_bleu(pds, gds)
    print("BLEU = %.2f, BLEU_1 = %.1f, BLEU_2 =%.1f, BLEU_3 = %.1f, BLEU_4 = %.1f (BP=%.3f, ratio=%.3f, hyp_len=%d, ref_len=%d)"%(bleu[0]*100, bleu[1]*100, bleu[2]*100, bleu[3]*100, bleu[4]*100, addition[0], addition[1], addition[2], addition[3]))

def bleubox(pds, gds):
    _len = len(pds)
    for i in range(_len):
        print(bleu(pds[i],gds[i])[1])

def auc(pds, gds):
    roc_auc_score =evaluate.load("roc_auc", "multilabel")
    results = roc_auc_score.compute(references=gds, prediction_scores=pds, average=None)
    print(results)

def meteor(pds, gds):
    meteor = evaluate.load('meteor')
    results = meteor.compute(predictions=pds, references=gds)
    print("meteor score: %.4f; "%(results['meteor']))


def precision(pds, gds):
    precision_metric = evaluate.load("precision")
    results = precision_metric.compute(references=gds, predictions=pds)
    print("precision score: %.4f; "%(results['precision']))



def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-metric', type = str, choices=['recall', 'auc', 'nltkbleu', 'meteor','precision','googlebleu', 'bleu','rouge','chrF','rouge'], default='bleu')
    parser.add_argument('-hyp', type=str, required=True) 
    parser.add_argument('-ref', type=str, required=True)
    opt = parser.parse_args()

    groundtruths = open(opt.ref, 'rb')
    gds = groundtruths.readlines()
    groundtruths.close()
    preds = open(opt.hyp,'rb')
    pds = preds.readlines()
    preds.close()    
    pds = [i.decode("utf-8") for i in pds]
    gds = [i.decode("utf-8") for i in gds]

    
    
    if opt.metric=='recall':
        recall(pds,gds)
    elif opt.metric=='chrF':
        chrF(pds, gds)
    elif opt.metric=='rouge':
        rouge(pds, gds)
        rougebox(pds, gds)
    elif opt.metric=='googlebleu':
        googlebleu(pds, gds)
    elif opt.metric=='bleu':
        bleu(pds, gds)
    elif opt.metric=='auc':
        auc(pds, gds)
    elif opt.metric=='precision':
        precision(pds, gds)
    elif opt.metric=='meteor':
        meteor(pds,gds)
    elif opt.metric=='nltkbleu':
        nltkbleu(pds,gds)

    

if __name__ == '__main__':
    main()