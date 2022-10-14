import argparse
from argparse import RawTextHelpFormatter
def parse_option():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--pf_hypothesis', type=str, default='all_data.txt')
    parser.add_argument('--model_name', type=str, default="DialoGPT-small")
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--min_len_generated', type=int, default=10)
    parser.add_argument('--max_len_generated', type=int, default=50)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--debug', type=bool, default=False)
    parser = argparse.ArgumentParser(
    description="Calculate BLEU scores for the input hypothesis and reference files")
parser.add_argument(
    "-hyp",
    nargs=1,
    dest="pf_hypothesis",
    type=str,
    help="The path of the hypothesis file.")
parser.add_argument(
    "-ref",
    nargs='+',
    dest="pf_references",
    type=str,
    help="The path of the references files.")
args = parser.parse_args()

    opt = parser.parse_args()
    return opt

def main():
    opt = parse_option()
