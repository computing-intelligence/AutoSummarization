import re
import jieba
from jieba.analyse import extract_tags, textrank



def cut(string): return ' '.join(jieba.cut(string))


def split_sentences(text, p='(：“|。”|？”|[。.，,？])', filter_p='\s+'):
    f_p = re.compile(filter_p)
    text = re.sub(f_p, '', text)
    pattern = re.compile(p)
    split = re.split(pattern, text)
    return split


def merge_sen_from_scores(text, score_fn, max_len=200):
    splited = split_sentences(text)
    ranked_sen = score_fn(text)
    selected_sen = set()

    temp_text = ''

    for sen, _ in ranked_sen:
        if len(temp_text) < max_len:
            temp_text += sen
            selected_sen.add(sen)

    summary_sen = []
    for sen, punc in zip(splited[::2], splited[1::2]):
        if sen in selected_sen:
            summary_sen.extend([sen, punc])

    return cut_tail(summary_sen)


def cut_tail(sen_list, fin_pun='。.？'):
    if sen_list[-1] in set(fin_pun):
        return sen_list
    else:
        return cut_tail(sen_list[:-2])


def extra_kw(text, n): return set(list(extract_tags(text, n))+list(textrank(text, n)))



