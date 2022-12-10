import os
import random
import re
import pickle

import lightgbm as lgb
import numpy as np
from scipy.spatial.distance import cosine

from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary

from bsbi import BSBIIndex
from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings

class Letor:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.documents = self.retrieve_documents()
        self.queries = self.retrieve_queries()
        self.traning_dataset, self.group_qid_count = self.make_training_data()
        self.dictionary, self.lsa_model = self.build_lsa(200)
        self.ranker = self.train_model()


    def retrieve_documents(self):
        documents = {}
        with open(os.path.join(self.data_dir, "train.docs")) as file:
            for line in file:
                doc_id, content = line.split("\t")
                documents[doc_id] = content.split()
        
        return documents

    def retrieve_queries(self):
        queries = {}
        with open(os.path.join(self.data_dir, "train.vid-desc.queries"), encoding="utf-8") as file:
            for line in file:
                q_id, content = line.split("\t")
                queries[q_id] = content.split()

        return queries

    def make_training_data(self):
        # melalui qrels, kita akan buat sebuah dataset untuk training
        # LambdaMART model dengan format
        #
        # [(query_text, document_text, relevance), ...]
        #
        # relevance awalnya bernilai 1, 2, 3 --> tidak perlu dinormalisasi
        # biarkan saja integer (syarat dari library LightGBM untuk
        # LambdaRank)
        #
        # relevance level: 3 (fully relevant), 2 (partially relevant), 1 (marginally relevant)
        import random

        NUM_NEGATIVES = 1

        q_docs_rel = {} # grouping by q_id terlebih dahulu
        with open(os.path.join(self.data_dir, "train.3-2-1.qrel")) as file:
            for line in file:
                q_id, _, doc_id, rel = line.split("\t")
                if (q_id in self.queries) and (doc_id in self.documents):
                    if q_id not in q_docs_rel:
                        q_docs_rel[q_id] = []
                    q_docs_rel[q_id].append((doc_id, int(rel)))

        # group_qid_count untuk model LGBMRanker
        group_qid_count = []
        dataset = []
        for q_id in q_docs_rel:
            docs_rels = q_docs_rel[q_id]
            group_qid_count.append(len(docs_rels) + NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                dataset.append((self.queries[q_id], self.documents[doc_id], rel))
            # tambahkan satu negative (random sampling saja dari documents)
            dataset.append((self.queries[q_id], random.choice(list(self.documents.values())), 0))

        return dataset, group_qid_count

    def build_lsa(self, num_latent_topics):
        # bentuk dictionary, bag-of-words corpus, dan kemudian Latent Semantic Indexing
        # dari kumpulan 3612 dokumen.

        dictionary = Dictionary()
        bow_corpus = [dictionary.doc2bow(doc, allow_update = True) for doc in self.documents.values()]
        model = LsiModel(bow_corpus, num_topics = num_latent_topics) # 200 latent topics

        return dictionary, model

    def vector_rep(self, text, num_latent_topics):
        rep = [topic_value for (_, topic_value) in self.lsa_model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == num_latent_topics else [0.] * num_latent_topics
    

    # kita ubah dataset menjadi terpisah X dan Y
    # dimana X adalah representasi gabungan query+document,
    # dan Y adalah label relevance untuk query dan document tersebut.
    # 
    # Bagaimana cara membuat representasi vector dari gabungan query+document?
    # cara simple = concat(vector(query), vector(document)) + informasi lain
    # informasi lain -> cosine distance & jaccard similarity antara query & doc

    def features(self, query, doc):
        v_q = self.vector_rep(query, 200)
        v_d = self.vector_rep(doc, 200)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [jaccard] + [cosine_dist]

    def separate_dataset(self):
        X = []
        Y = []
        for (query, doc, rel) in self.traning_dataset:
            X.append(self.features(query, doc))
            Y.append(rel)

        # ubah X dan Y ke format numpy array
        X = np.array(X)
        Y = np.array(Y)

        return X,Y

    def train_model(self):

        ranker = lgb.LGBMRanker(
                    objective="lambdarank",
                    boosting_type = "gbdt",
                    n_estimators = 100,
                    importance_type = "gain",
                    metric = "ndcg",
                    num_leaves = 40,
                    learning_rate = 0.02,
                    max_depth = -1)
        
        X, Y = self.separate_dataset()

        ranker.fit(X, Y,
                   group = self.group_qid_count,
                   verbose = 10)
        
        return ranker

    def predict(self, query):
        
        cur_path = os.path.dirname(__file__)
        print(cur_path)
        BSBI_instance = BSBIIndex(data_dir = os.path.join(cur_path, 'collection'), \
                          postings_encoding = VBEPostings, \
                          output_dir = os.path.join(cur_path, 'index'))

        top100_bm25 = BSBI_instance.retrieve_bm25(query=query, k1=1.5, b=0.75, k=100)
        # for (score, doc) in top100_bm25:
        #     doc_id = int(re.search(r'.*\\(.*)\.txt', doc).group(1))
        #     print(f"D{doc_id} : {score:.2f}")

        docs = []
        for (score, doc) in top100_bm25:
            doc_id = int(re.search(r'.*\\(.*)\.txt', doc).group(1))
            with open(os.path.join(cur_path, doc), 'r') as file:
                docs.append((str(doc_id), file.read()))

        X_unseen = []
        for doc_id, doc in docs:
            X_unseen.append(self.features(query.split(), doc.split()))

        X_unseen = np.array(X_unseen)
        scores = self.ranker.predict(X_unseen)

        did_scores = [x for x in zip([did for (did, _) in docs], scores, [doc_text for (_, doc_text) in docs])]
        sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)

        result = {}
        counter = 1
        for (did, score, doc_text) in sorted_did_scores:
            result[counter] = {"docId": did, "docText": doc_text}
            counter += 1
        return result

def main():

    cur_path = os.path.dirname(__file__)
    # letor = Letor(os.path.join(cur_path, 'nfcorpus'))

    query = "how much cancer risk can be avoided through lifestyle change ?"

    # docs =[("D1", "dietary restriction reduces insulin-like growth factor levels modulates apoptosis cell proliferation tumor progression num defici pubmed ncbi abstract diet contributes one-third cancer deaths western world factors diet influence cancer elucidated reduction caloric intake dramatically slows cancer progression rodents major contribution dietary effects cancer insulin-like growth factor igf-i lowered dietary restriction dr humans rats igf-i modulates cell proliferation apoptosis tumorigenesis mechanisms protective effects dr depend reduction multifaceted growth factor test hypothesis igf-i restored dr ascertain lowering igf-i central slowing bladder cancer progression dr heterozygous num deficient mice received bladder carcinogen p-cresidine induce preneoplasia confirmation bladder urothelial preneoplasia mice divided groups ad libitum num dr num dr igf-i igf-i/dr serum igf-i lowered num dr completely restored igf-i/dr-treated mice recombinant igf-i administered osmotic minipumps tumor progression decreased dr restoration igf-i serum levels dr-treated mice increased stage cancers igf-i modulated tumor progression independent body weight rates apoptosis preneoplastic lesions num times higher dr-treated mice compared igf/dr ad libitum-treated mice administration igf-i dr-treated mice stimulated cell proliferation num fold hyperplastic foci conclusion dr lowered igf-i levels favoring apoptosis cell proliferation ultimately slowing tumor progression mechanistic study demonstrating igf-i supplementation abrogates protective effect dr neoplastic progression"), 
    #     ("D2", "study hard as your blood boils"), 
    #     ("D3", "processed meats risk childhood leukemia california usa pubmed ncbi abstract relation intake food items thought precursors inhibitors n-nitroso compounds noc risk leukemia investigated case-control study children birth age num years los angeles county california united states cases ascertained population-based tumor registry num num controls drawn friends random-digit dialing interviews obtained num cases num controls food items principal interest breakfast meats bacon sausage ham luncheon meats salami pastrami lunch meat corned beef bologna hot dogs oranges orange juice grapefruit grapefruit juice asked intake apples apple juice regular charcoal broiled meats milk coffee coke cola drinks usual consumption frequencies determined parents child risks adjusted risk factors persistent significant associations children's intake hot dogs odds ratio num num percent confidence interval ci num num num hot dogs month trend num fathers intake hot dogs num ci num num highest intake category trend num evidence fruit intake provided protection results compatible experimental animal literature hypothesis human noc intake leukemia risk potential biases data study hypothesis focused comprehensive epidemiologic studies warranted"), 
    #     ("D4", "long-term effects calorie protein restriction serum igf num igfbp num concentration humans summary reduced function mutations insulin/igf-i signaling pathway increase maximal lifespan health span species calorie restriction cr decreases serum igf num concentration num protects cancer slows aging rodents long-term effects cr adequate nutrition circulating igf num levels humans unknown report data long-term cr studies num num years showing severe cr malnutrition change igf num igf num igfbp num ratio levels humans contrast total free igf num concentrations significantly lower moderately protein-restricted individuals reducing protein intake average num kg num body weight day num kg num body weight day num weeks volunteers practicing cr resulted reduction serum igf num num ng ml num num ng ml num findings demonstrate unlike rodents long-term severe cr reduce serum igf num concentration igf num igfbp num ratio humans addition data provide evidence protein intake key determinant circulating igf num levels humans suggest reduced protein intake important component anticancer anti-aging dietary interventions"), 
    #     ("D5", "cancer preventable disease requires major lifestyle abstract year num million americans num million people worldwide expected diagnosed cancer disease commonly believed preventable num num cancer cases attributed genetic defects remaining num num roots environment lifestyle lifestyle factors include cigarette smoking diet fried foods red meat alcohol sun exposure environmental pollutants infections stress obesity physical inactivity evidence cancer-related deaths num num due tobacco num num linked diet num num due infections remaining percentage due factors radiation stress physical activity environmental pollutants cancer prevention requires smoking cessation increased ingestion fruits vegetables moderate alcohol caloric restriction exercise avoidance direct exposure sunlight minimal meat consumption grains vaccinations regular check-ups review present evidence inflammation link agents/factors cancer agents prevent addition provide evidence cancer preventable disease requires major lifestyle")]

    # sekedar pembanding, ada bocoran: D3 & D5 relevant, D1 & D4 partially relevant, D2 tidak relevan

    # bentuk ke format numpy array
    # X_unseen = []
    # for doc_id, doc in docs:
    #     X_unseen.append(letor.features(query.split(), doc.split()))

    # X_unseen = np.array(X_unseen)

    # scores = letor.ranker.predict(X_unseen)
    # print(scores)

    # BSBI_instance = BSBIIndex(data_dir = 'collection', \
    #                       postings_encoding = VBEPostings, \
    #                       output_dir = 'index')

    # top100_bm25 = BSBI_instance.retrieve_bm25(query=query, k1=1.5, b=0.75, k=100)
    # # for (score, doc) in top100_bm25:
    # #     doc_id = int(re.search(r'.*\\(.*)\.txt', doc).group(1))
    # #     print(f"D{doc_id} : {score:.2f}")

    # docs = []
    # for (score, doc) in top100_bm25:
    #     doc_id = int(re.search(r'.*\\(.*)\.txt', doc).group(1))
    #     with open(doc, 'r') as file:
    #         docs.append((str(doc_id), file.read()))

    # X_unseen = []
    # for doc_id, doc in docs:
    #     X_unseen.append(letor.features(query.split(), doc.split()))

    # X_unseen = np.array(X_unseen)
    # scores = letor.ranker.predict(X_unseen)

    # did_scores = [x for x in zip([did for (did, _) in docs], scores, [doc_text for (_, doc_text) in docs])]
    # sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)

    sorted_did_scores = None
    with open(os.path.join(cur_path, 'letor.pkl'), 'rb') as file:
        letor = pickle.load(file)
        sorted_did_scores = letor.predict(query)


    print("query        :", query)
    print("SERP/Ranking :")
    # for (did, score, doc_text) in sorted_did_scores:
    #     print(f"D{did} : {score:.2f}")
    print(sorted_did_scores)

if __name__ == '__main__':
    main()