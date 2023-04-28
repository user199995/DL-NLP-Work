import random
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# 从语料库中随机抽取200个段落
def extract_paragraphs(corpus_path, num_paragraphs=200, min_words=500):
    paragraphs = []
    with open(corpus_path, 'r') as f:
        texts = f.readlines()
        while len(paragraphs) < num_paragraphs:
            text = random.choice(texts)
            # 分割文本为段落
            text = re.sub('\n+', '\n', text)
            text = re.sub('\r+', '\r', text)
            text = re.sub('[\r\n]+', '\n', text)
            text = text.strip()
            if not text:
                continue
            for paragraph in text.split('\n'):
                # 过滤掉过短的段落
                if len(paragraph.split()) < min_words:
                    continue
                paragraphs.append(paragraph)
                if len(paragraphs) == num_paragraphs:
                    break
    return paragraphs

# 为每个段落分配标签（即所属的小说）
def assign_labels(paragraphs, labels):
    labeled_paragraphs = []
    for paragraph in paragraphs:
        for label in labels:
            if label in paragraph.lower():
                labeled_paragraphs.append((paragraph, label))
                break
    return labeled_paragraphs

# 对文本进行预处理，包括分词、去除停用词等
def preprocess(text):
    # 分句
    sentences = sent_tokenize(text)
    # 分词
    words = []
    for sentence in sentences:
        words.extend(nltk.word_tokenize(sentence))
    # 去除停用词
    stop_words = set(stopwords.words('english'))
    words = [w.lower() for w in words if w.lower() not in stop_words]
    return words
from gensim import corpora, models

# 训练LDA模型
def train_lda_model(paragraphs, num_topics):
    texts = [preprocess(paragraph) for paragraph, _ in paragraphs]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
    return lda_model

# 表示段落为主题分布
def get_topic_distribution(paragraph, lda_model):
    text = preprocess(paragraph)
    bow = lda_model.id2word.doc2bow(text)
    topic_distribution = lda_model[bow]
    return topic_distribution
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# 将段落表示为特征向量
def vectorize_paragraphs(paragraphs, vectorizer_type='count', ngram_range=(1,1)):
    texts = [preprocess(paragraph) for paragraph, _ in paragraphs]
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer(ngram_range=ngram_range)
    elif vectorizer_type =='tfidf':
vectorizer = TfidfVectorizer(ngram_range=ngram_range)
else:
raise ValueError('Invalid vectorizer type')
features = vectorizer.fit_transform([' '.join(text) for text in texts])
return features
def train_and_evaluate_classifier(features, labels):
# 将数据划分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
# # 训练逻辑回归分类器
# classifier = LogisticRegression()
# classifier.fit(X_train, y_train)
# # 在测试集上进行预测和评估
# y_pred = classifier.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='weighted')
# recall = recall_score(y_test, y_pred, average='weighted')
# f1 = f1_score(y_test, y_pred, average='weighted')
# return accuracy, precision, recall, f1
corpus_path = 'corpus.txt'
labels = ['novel1', 'novel2', 'novel3', 'novel4']
num_topics_list = [5, 10, 15, 20]
vectorizer_types = ['count', 'tfidf']
ngram_ranges = [(1, 1), (1, 2), (2, 2)]

# 抽取段落并分配标签
paragraphs = extract_paragraphs(corpus_path)
labeled_paragraphs = assign_labels(paragraphs, labels)

# 对不同参数进行实验
for num_topics in num_topics_list:
    # 训练LDA模型
    lda_model = train_lda_model(labeled_paragraphs, num_topics)
    for vectorizer_type in vectorizer_types:
        for ngram_range in ngram_ranges:
            # 将段落表示为特征向量
            features = vectorize_paragraphs(labeled_paragraphs, vectorizer_type, ngram_range)
            # 分类器训练和评估
            accuracy, precision, recall, f1 = train_and_evaluate_classifier(features,
                                                                            [label for _, label in labeled_paragraphs])
            print(
                f'num_topics={num_topics}, vectorizer_type={vectorizer_type}, ngram_range={ngram_range}, accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}')