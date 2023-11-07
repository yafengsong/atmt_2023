merge_table_path = './example/subword_nmt.voc'

from bpe import load_subword_nmt_table, BpeOnlineTokenizer
import subprocess

merge_table = load_subword_nmt_table(merge_table_path)

subword_nmt_tokenizer = BpeOnlineTokenizer(bpe_dropout_rate=0.1, merge_table=merge_table)


# BPE tokenizing training datasets (en and fr)
with open('data/en-fr/raw/raw_train.en', 'r') as exist_file, open('data/en-fr/raw/train.en', 'w') as new_file:
    for line in exist_file:
        bpe_sentence = subword_nmt_tokenizer(line, sentinels=['', '</w>'], regime='end', bpe_symbol='@@')
        new_file.write(bpe_sentence)

with open('data/en-fr/raw/raw_train.fr', 'r') as exist_file, open('data/en-fr/raw/train.fr', 'w') as new_file:
    for line in exist_file:
        bpe_sentence = subword_nmt_tokenizer(line, sentinels=['', '</w>'], regime='end', bpe_symbol='@@')
        new_file.write(bpe_sentence)

# Execute preprosessing (binerize)
bash_script = './preprocess_data.sh'
subprocess.run(['chmod', '+x', bash_script])
subprocess.run([bash_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
