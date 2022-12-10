import os

from bsbi import BSBIIndex
from compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
cur_path = os.path.dirname(__file__)
BSBI_instance = BSBIIndex(data_dir = os.path.join(cur_path, 'collection'), \
                          postings_encoding = VBEPostings, \
                          output_dir = os.path.join(cur_path, 'index'))

queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]



for query in queries:
    print("Query  : ", query)
    print("Results:")
    print("TF-IDF")
    for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = 10):
        print(f"{doc:30} {score:>.3f}")
    print()
    print("BM25")
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k1 = 1.5, b = 0.75, k = 10):
        print(f"{doc:30} {score:>.3f}")
    print()

# "alkylated with radioactive iodoacetate"