import math
import sys
import time

import metapy
import pytoml

from scipy import stats

class InL2Ranker(metapy.index.RankingFunction):
    """
    Create a new ranking function in Python that can be used in MeTA.
    """
    def __init__(self, param=1.0):
        self.param = param
        # You *must* call the base class constructor here!
        super(InL2Ranker, self).__init__()

    def score_one(self, sd):
        """
        You need to override this function to return a score for a single term.
        For fields available in the score_data sd object,
        @see https://meta-toolkit.org/doxygen/structmeta_1_1index_1_1score__data.html
        """
        tfn = sd.doc_term_count * (math.log((1 + (sd.avg_dl/sd.doc_size)), 2))

        score = sd.query_term_weight * (tfn/(tfn + self.param)) * (math.log((sd.num_docs + 1)/(sd.corpus_term_count + 0.5), 2))

        #return (self.param + sd.doc_term_count) / (self.param * sd.doc_unique_terms + sd.doc_size)
        return score

def load_ranker_bm25(cfg_file):
    """
    Use this function to return the Ranker object to evaluate, e.g. return InL2Ranker(some_param=1.0)
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index. You can ignore this for MP2.
    """
    #return metapy.index.JelinekMercer(0.72)
    #return metapy.index.DirichletPrior(158)
    #return metapy.index.AbsoluteDiscount(0.7)
    #return metapy.index.PivotedLength()
    return metapy.index.OkapiBM25(2.0, 0.70, 500.0)

def load_ranker_inl2(cfg_file):
    """
    Use this function to return the Ranker object to evaluate, e.g. return InL2Ranker(some_param=1.0)
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index. You can ignore this for MP2.
    """

    return InL2Ranker(param=2.0)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} config.toml".format(sys.argv[0]))
        sys.exit(1)

    cfg = sys.argv[1]
    print('Building or loading index...')
    idx = metapy.index.make_inverted_index(cfg)
    ev_bm25 = metapy.index.IREval(cfg)
    ev_inl2 = metapy.index.IREval(cfg)

    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin)

    query_cfg = cfg_d['query-runner']
    if query_cfg is None:
        print("query-runner table needed in {}".format(cfg))
        sys.exit(1)

    start_time = time.time()
    top_k = 10
    query_path = query_cfg.get('query-path', 'queries.txt')
    query_start = query_cfg.get('query-id-start', 0)
    query = metapy.index.Document()

    ranker_bm25 = load_ranker_bm25(cfg)
    ranker_inl2 = load_ranker_inl2(cfg)
    list_avg_p_bm25 = []
    list_avg_p_inl2 = []

    print('Running queries')
    with open(query_path) as query_file:
        for query_num, line in enumerate(query_file):
            query.content(line.strip())
            results_bm25 = ranker_bm25.score(idx, query, top_k)
            results_inl2 = ranker_inl2.score(idx, query, top_k)
            avg_p_bm25 = ev_bm25.avg_p(results_bm25, query_start + query_num, top_k)
            avg_p_inl2 = ev_inl2.avg_p(results_inl2, query_start + query_num, top_k)
            list_avg_p_bm25.append(avg_p_bm25)
            list_avg_p_inl2.append(avg_p_inl2)
            #print("Query {} average precision BM25: {}".format(query_num + 1, avg_p_bm25))
            #print("Query {} average precision INL2: {}".format(query_num + 1, avg_p_inl2))
    with open('bm25.avg_p.txt', 'w') as f:
        for item in list_avg_p_bm25:
            f.write("%s\n" % item)
    with open('inl2.avg_p.txt', 'w') as f:
        for item in list_avg_p_inl2:
            f.write("%s\n" % item)
    p = stats.ttest_rel(list_avg_p_bm25, list_avg_p_inl2)
    print("p: {}".format(p))
    with open('significance.txt', 'w') as f:
        f.write("%s" % p.pvalue)
    print("Mean average precision BM25: {}".format(ev_bm25.map()))
    print("Mean average precision INL2: {}".format(ev_inl2.map()))
    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))
