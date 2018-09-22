from utils import *
import networkx
import sys
import warnings

warnings.filterwarnings('ignore')


def get_summarization_by_textrank(text, max_len=200):return merge_sen_from_scores(text, text_rank, max_len)


def text_rank(text, window=5):
    graph = get_sentence_graph(text, window)
    scored_sen = networkx.pagerank(graph)
    ranked_sen = sorted(scored_sen.items(), key=lambda x: x[-1], reverse=True)
    return ranked_sen


def get_sentence_graph(text, window=5):
    sen_graph = networkx.graph.Graph()
    sub_sens = split_sentences(text)[::2]

    for i, sen in enumerate(sub_sens):
        for ii in range(i - window, i + window + 1):
            if ii >= 0 and ii < len(sub_sens):
                edge = (sen, sub_sens[ii])
                sen_graph.add_edges_from([edge])
    return sen_graph


with open(sys.argv[1], 'r',encoding='utf-8') as f:
    t = f.readlines()[0]
    summary = ''.join(get_summarization_by_textrank(t))

print(summary)

if summary is not None:
    print('save in working dir?Y/N')
    if input() in ('Y', 'y'):
        with open(sys.argv[1][:-4]+'summ.txt','w') as f:
            f.write(summary)
        print('finished!')
