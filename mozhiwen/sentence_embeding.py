from utils import *
from gensim.models import FastText
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import numpy as np
import pickle
from scipy.spatial.distance import cosine
from functools import partial
import warnings

warnings.filterwarnings('ignore')


word_frequency_path = './word_fre.pkl'
model_path = './fasttext_model/news_token.model'

with open(word_frequency_path, 'rb') as f:
    word_frequency = pickle.load(f)

model = FastText.load(model_path)


def sif_sentence_embedding(text, alpha=1e-4):
    global word_frequency

    max_fre = max(word_frequency.values())
    sen_vec = np.zeros_like(model.wv['测试'])
    words = cut(text).split()
    words = [w for w in words if w in model]

    for w in words:
        fre = word_frequency.get(w, max_fre)
        weight = alpha / (fre + alpha)
        sen_vec += weight * model.wv[w]

    sen_vec /= len(words)
    # skip SVD
    return sen_vec


def get_corr(text, sub_sen_vec, embed_fn):
    if isinstance(text, list): text = ' '.join(text)

    topic_vec = embed_fn(text)

    score = cosine(topic_vec, sub_sen_vec)

    return score


get_corr_sif = partial(get_corr, embed_fn=sif_sentence_embedding)


def lda_score(text, sub_sen_vec):
    plain_text = list(map(lambda x:cut(x).split(),filter(lambda x:x!='',split_sentences(text)[::2])))
    common_dict = Dictionary(plain_text)
    common_corpus = [common_dict.doc2bow(t) for t in plain_text]

    no_topics = int(len(plain_text)/2)
    no_words = int(len(text)/no_topics/2)

    lda = LdaModel(common_corpus, num_topics=no_topics)

    topic_list = []
    for i in range(no_topics):
        topic_list.append([common_dict[t[0]] for t in lda.get_topic_terms(i, no_words)])

    score_list = []
    for topic in topic_list:
        score_list.append(get_corr_sif(topic, sub_sen_vec))
    return max(score_list)


def get_corr_title(text, title, embed_fn, c_w=0.6, ti_w=0.2, to_w=0.2):
    if isinstance(text, list): text = ' '.join(text)

    sub_sentences = split_sentences(text)[::2]
    sen_vec = embed_fn(text)
    title_vec = embed_fn(title)

    corr_score = {}

    for sen in sub_sentences:
        sub_sen_vec = embed_fn(sen)
        corr_score[sen] = c_w * cosine(sen_vec, sub_sen_vec) + ti_w * cosine(title_vec, sub_sen_vec) + to_w * lda_score(text, sub_sen_vec)

    return sorted(corr_score.items(), key=lambda x: x[1], reverse=True)


get_corr_title_sif = partial(get_corr_title, embed_fn=sif_sentence_embedding)


def knn_rescore(sub_sentences, ranking_sentences, windows=3):
    origin_scores = {k: s for k, s in ranking_sentences}
    rescores = {}

    for i, sen in enumerate(sub_sentences):
        temp_list = []
        for ii in range(i - windows, i + windows + 1):
            if ii >= 0 and ii < len(sub_sentences):
                temp_list.append(origin_scores[sub_sentences[ii]])

        rescores[sen] = np.mean(temp_list)

    rescores_sort = sorted(rescores.items(), key=lambda x: x[1], reverse=True)
    return rescores_sort


def kw_rescore(keywords, ranking_sentences, ex_weigth=0.05, decay=0.5):
    origin_scores = {k: s for k, s in ranking_sentences}
    rescores = {}

    for sen, score in origin_scores.items():
        n = 0
        rescores[sen] = origin_scores[sen]
        for word in cut(sen).split():
            if word in keywords:
                rescores[sen] += origin_scores[sen] * ex_weigth * (decay ** n)

    rescores_sort = sorted(rescores.items(), key=lambda x: x[1], reverse=True)
    return rescores_sort


def position_rescore(sub_sentences, ranking_sentences, ex_weight=0.15):
    origin_scores = {k: s for k, s in ranking_sentences}
    rescores = {}

    for sen, score in origin_scores.items():
        if sen == sub_sentences[0] or sen == sub_sentences[-1]:
            rescores[sen] = origin_scores[sen] * (1 + ex_weight)
        else:
            rescores[sen] = origin_scores[sen]

    rescores_sort = sorted(rescores.items(), key=lambda x: x[1], reverse=True)
    return rescores_sort


def merge_sen_from_scores_opt(text, title, score_fn, max_len=200):
    splited = split_sentences(text)
    kw = extra_kw(text, int(len(text)/10))
    ranked_sen = score_fn(text, title)
    ranked_sen = kw_rescore(kw, ranked_sen, 0.05, 1)
    ranked_sen = position_rescore(splited[::2], ranked_sen, 0.15)
    ranked_sen = knn_rescore(splited[::2], ranked_sen,windows=int(len(splited[::2])/5))

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


def get_summarization_by_senemb(text,title,max_len=200): return merge_sen_from_scores_opt(text, title, get_corr_title_sif,max_len)


if __name__ == '__main__':

    content = '''虽然至今夏普智能手机在市场上无法排得上号，已经完全没落，并于 2013 年退出中国市场，但是今年 3 月份官方突然宣布回归中国，预示着很快就有夏普新机在中国登场了。那么，第一款夏普手机什么时候登陆中国呢？又会是怎么样的手机呢？\r\n近日，一款型号为 FS8016 的夏普神秘新机悄然出现在 GeekBench 的跑分库上。从其中相关信息了解到，这款机子并非旗舰定位，所搭载的是高通骁龙 660 处理器，配备有 4GB 的内存。骁龙 660 是高通今年最受瞩目的芯片之一，采用 14 纳米工艺，八个 Kryo 260 核心设计，集成 Adreno 512 GPU 和 X12 LTE 调制解调器。\r\n当前市面上只有一款机子采用了骁龙 660 处理器，那就是已经上市销售的 OPPO R11。骁龙 660 尽管并非旗舰芯片，但在多核新能上比去年骁龙 820 强，单核改进也很明显，所以放在今年仍可以让很多手机变成高端机。不过，由于 OPPO 与高通签署了排他性协议，可以独占两三个月时间。\r\n考虑到夏普既然开始测试新机了，说明只要等独占时期一过，夏普就能发布骁龙 660 新品了。按照之前被曝光的渲染图了解，夏普的新机核心竞争优势还是全面屏，因为从 2013 年推出全球首款全面屏手机 EDGEST 302SH 至今，夏普手机推出了多达 28 款的全面屏手机。\r\n在 5 月份的媒体沟通会上，惠普罗忠生表示：“我敢打赌，12 个月之后，在座的各位手机都会换掉。因为全面屏时代的到来，我们怀揣的手机都将成为传统手机。”\r\n'''
    title ='''配骁龙660 全面屏鼻祖夏普新机酝酿中'''
    print(''.join(get_summarization_by_senemb(content, title, 300)))
    print('test pass!')