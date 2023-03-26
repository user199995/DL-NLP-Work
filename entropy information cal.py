import os
import math
import re
import requests
import chardet
import jieba

def count_chinese_chars(file_path):
    # read the file contents
    with open(file_path, 'r', encoding='gb18030') as f:
        text = f.read()

    # count the number of Chinese characters
    num_chars = 0
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            num_chars += 1

    return num_chars

def entropy(file_path, is_word_entropy=False, exclude_words=None):
    # read the file contents
    with open(file_path, 'r', encoding='gb18030') as f:
        text = f.read()

    # count the frequency of each character or word
    freq = {}
    if is_word_entropy:
        words = re.findall(r'\b\w+\b', text)
        for word in words:
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1
    else:
        for char in text:
            if char in freq:
                freq[char] += 1
            else:
                freq[char] = 1

    # calculate the probability of each character or word
    total_chars_or_words = len(text) if not is_word_entropy else len(words)
    prob = {}
    for char_or_word, count in freq.items():
        if exclude_words and char_or_word in exclude_words:
            continue
        prob[char_or_word] = count / total_chars_or_words

    # calculate the entropy
    entropy = 0
    for prob in prob.values():
        entropy += prob * math.log(prob, 2)

    return -entropy


def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf_8') as f:
        text = f.read()
    words = jieba.cut(text)
    return [word for word in words]



# specify the directory containing the TXT files
directory = r'C:\Users\31217\Desktop\corpus'
stop_words = r'C:\Users\31217\Desktop\stop words\停词.txt'

exclude_words = read_txt_file(stop_words)

# iterate over the files in the directory

for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        char_entropy = entropy(file_path, exclude_words=exclude_words)
        word_entropy = entropy(file_path, is_word_entropy=True, exclude_words=exclude_words)
        num_chars = count_chinese_chars(file_path)
        print(
            f'{filename}: character entropy = {char_entropy}, word entropy = {word_entropy}, num Chinese chars = {num_chars}')
