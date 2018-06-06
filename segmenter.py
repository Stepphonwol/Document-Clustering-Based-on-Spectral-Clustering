# encoding=utf-8
import zhon.hanzi
import re
import string
import jieba
import gensim
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

class Transformer:
    def segmenter(self):
        '''
        seg = StanfordSegmenter(
            path_to_jar="/Users/lumindoec/Downloads/stanford-segmenter-2018-02-27/stanford-segmenter.jar",
            path_to_dict="/Users/lumindoec/Downloads/stanford-segmenter-2018-02-27/data/dict-chris6.ser.gz",
            path_to_model="/Users/lumindoec/Downloads/stanford-segmenter-2018-02-27/data/pku.gz",
            path_to_sihan_corpora_dict="/Users/lumindoec/Downloads/stanford-segmenter-2018-02-27/data",
            java_class='edu.stanford.nlp.ie.crf.CRFClassifier')
        sent = u'这是斯坦福中文分词器测试'
        print(seg.segment(sent))
        '''
        print("Fetching input text...")
        #path = "/Users/lumindoec/PyCharmProjects/doc_clustering/movie"
        #path = "/Users/lumindoec/PyCharmProjects/doc_clustering/info"
        path = "/Users/lumindoec/PyCharmProjects/doc_clustering/m_movie"
        self.seg_list = []
        self.name_list = []
        for i in range(1, 1019):
            f = open(path + "/" + str(i) + ".txt", "r+")
            print(f)
            raw_name = f.readline()
            print(raw_name)
            # filtering outliers
            #if i == 62:
                #continue
            self.name_list.append(raw_name)
            raw_file = f.readline()
            print(raw_file)
            print("Segmenting!!!")
            # remove the \n in the end
            if(raw_file[len(raw_file) - 1] == '\n'):
                raw_file = raw_file[:len(raw_file) - 1]
            raw_file = self.remove_letters(raw_file)
            raw_list = jieba.lcut(raw_file, cut_all=False)
            raw_seg = " ".join(raw_list)
            print(self.remove_punctuation(raw_seg))
            #print(raw_seg)
            self.seg_list.append(self.remove_punctuation(raw_seg))
            # when no need to do word segmenting
            #self.seg_list.append(raw_file)
            '''
            f.readline()
            raw_url = f.readline()
            print(raw_url)
            self.url_list.append(raw_url)
            '''
            f.close()
        print(len(self.name_list))
        print(len(self.seg_list))
        input()

    def remove_letters(self, text):
        return re.sub('[a-zA-Z]', '', text)

    def remove_punctuation(self, text):
        # remove chinese punctuation
        h_regex = re.compile('[%s]' % zhon.hanzi.punctuation)
        # remove non-chinese punctuation
        p_regex = re.compile('[%s]' % re.escape(string.punctuation))
        return p_regex.sub('', h_regex.sub('', text))

    def labeling(self):
        for index, sentence in enumerate(self.seg_list):
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.to_unicode(sentence).split(), [index])

    def numerizer(self, learning_rate):
        ''' tf-idf numerizer
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(self.seg_list))
        print(vectorizer.get_feature_names())
        self.vector_space = tfidf.toarray()
        print(self.vector_space)
        return self.vector_space, self.name_list
        '''
        # doc2vec numerizer
        train_src = list(self.labeling())
        model = gensim.models.doc2vec.Doc2Vec(window=8, min_count=5, alpha=learning_rate, min_alpha=0.001)
        model.build_vocab(train_src)
        model.train(train_src, total_examples=model.corpus_count, epochs=1000)
        print(len(model.docvecs))
        print(model.docvecs.doctag_syn0)
        return model.docvecs.doctag_syn0, self.name_list