from cmath import log
import os
import pickle
import contextlib
import heapq
import time
import math
import operator
from nltk.stem.snowball import SnowballStemmer
import spacy
import sys

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings
from tqdm import tqdm

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []
        self.avg_doc_length = self.get_avg_doc_length()

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""
        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        # TODO
        td_pairs = []
        sp = spacy.load('en_core_web_sm')
        stemmer = SnowballStemmer(language='english')
        doc_list = os.listdir(os.path.join(self.data_dir, block_dir_relative))

        for doc in doc_list:
            with open(os.path.join(self.data_dir, block_dir_relative, doc)) as f:
                doc_id = self.doc_id_map.__getitem__(f.name)
                doc_string = sp(f.read())
                for sentence in doc_string.sents:
                    for word in sentence:
                        if (word.is_stop or word.is_punct or word.is_space):
                            continue
                        token = stemmer.stem(word.text)
                        term_id = self.term_id_map.__getitem__(token)
                        td_pairs.append((term_id, doc_id))
        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        # TODO
        term_dict = {}

        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = (dict())
            if doc_id not in term_dict[term_id]:
                term_dict[term_id][doc_id] = 0

            term_dict[term_id][doc_id] += 1

        for term_id in sorted(term_dict.keys()):
            sorted_postings_tf = dict(sorted(term_dict[term_id].items()))
            index.append(term_id, list(sorted_postings_tf.keys()), list(sorted_postings_tf.values()))


    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        self.load()
        spa = spacy.load('en_core_web_sm')
        stemmer = SnowballStemmer(language='english')

        query_words = spa(query)
        clean_query = []
        for word in query_words:
            if (word.is_stop or word.is_punct or word.is_space):
                continue
            stemmed_word = stemmer.stem(word.text)
            clean_query.append(stemmed_word)

        result = {}
        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as index:
            for word in clean_query:
                word_id = self.term_id_map.__getitem__(word)
                if (word_id not in index.postings_dict):
                    continue
                postings_list, tf_list = index.get_postings_list(word_id)
                for doc_id, tf in zip(postings_list, tf_list):
                    if doc_id not in result:
                        result[doc_id] = 0

                    df = index.postings_dict[word_id][1]
                    idf = math.log10(len(index.doc_length) / df)
                    if (tf > 0):
                        w_tf = 1 + math.log10(tf)
                        result[doc_id] += w_tf * idf

        sorted_result = dict(sorted(result.items(), key=operator.itemgetter(1), reverse=True))
        top_10 = []

        for doc_id, score in list(sorted_result.items())[:10]:
            doc_name = self.doc_id_map.__getitem__(doc_id)
            top_10.append((score, doc_name))

        return top_10

    def retrieve_bm25(self, query, k1, b, k=100):

        self.load()
        spa = spacy.load('en_core_web_sm')
        stemmer = SnowballStemmer(language='english')

        query_words = spa(query)
        clean_query = []
        for word in query_words:
            if (word.is_stop or word.is_punct or word.is_space):
                continue
            stemmed_word = stemmer.stem(word.text)
            clean_query.append(stemmed_word)
        
        result = {}
        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as index:
            for word in clean_query:
                word_id = self.term_id_map.__getitem__(word)
                if (word_id not in index.postings_dict):
                    continue
                postings_list, tf_list = index.get_postings_list(word_id)
                for doc_id, tf in zip(postings_list, tf_list):
                    if doc_id not in result:
                        result[doc_id] = 0

                    df = index.postings_dict[word_id][1]
                    idf = math.log10(len(index.doc_length) / df)
                    if (tf > 0):
                        B = (   (1-b) + (b * (index.doc_length[doc_id] / self.avg_doc_length))   )
                        w_tf = (   ((k1 + 1) * tf) / (k1 * B + tf)   )
                        result[doc_id] += idf * w_tf

        sorted_result = dict(sorted(result.items(), key=operator.itemgetter(1), reverse=True))
        top_10 = []

        for doc_id, score in list(sorted_result.items())[:100]:
            doc_name = self.doc_id_map.__getitem__(doc_id)
            top_10.append((score, doc_name))

        return top_10

    def get_avg_doc_length(self):
        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as index:
            total_doc_length = 0
            total_doc = len(index.doc_length)
            for length in index.doc_length.values():
                total_doc_length += length

            return total_doc_length / total_doc

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":

    cur_path = os.path.dirname(__file__)
    BSBI_instance = BSBIIndex(data_dir = os.path.join(cur_path, 'collection'), \
                              postings_encoding = VBEPostings, \
                              output_dir = os.path.join(cur_path, 'index'))
    BSBI_instance.index() # memulai indexing!
