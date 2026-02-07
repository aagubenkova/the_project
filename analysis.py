import spacy
import pandas as pd
from collections import Counter

#  модель
nlp = spacy.load("en_core_web_sm")


gospels = [
    ("Matthew", "matthew.txt"),
    ("Mark", "mark.txt"),
    ("Luke", "luke.txt"),
    ("John", "john.txt")
]


def analyze_text(text):
    doc = nlp(text)

    pos_counter = Counter()
    sentence_lengths = []
    imperatives = 0

    #  части речи и императивы
    for token in doc:
        if token.is_alpha:
            pos_counter[token.pos_] += 1
            if token.pos_ == "VERB" and token.tag_ == "VB":
                imperatives += 1

    #  длина предложений
    for sent in doc.sents:
        sentence_lengths.append(len(sent))

    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)

    return {
        "NOUN": pos_counter["NOUN"],
        "VERB": pos_counter["VERB"],
        "ADJ": pos_counter["ADJ"],
        "PRON": pos_counter["PRON"],
        "Avg sentence length": round(avg_sentence_length, 2),
        "Imperatives": imperatives
    }

# Анализируем все Евангелия
rows = []

for gospel_name, file_name in gospels:
    with open(file_name, "r", encoding="utf-8") as f:
        text = f.read()

    result = analyze_text(text)
    result["Gospel"] = gospel_name
    rows.append(result)


df = pd.DataFrame(rows)
df = df.set_index("Gospel")

print(df)

import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk import word_tokenize
file_path = ["john.txt", "luke.txt", "mark.txt", "matthew.txt"]
all_text = ""
for filename in file_path:
    with open(filename, 'r', encoding='utf-8') as file:
        all_text += file.read() + " "  # объединяем тексты

tokens = word_tokenize(all_text.lower())
print(tokens[:20])



from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(text)
print(sentences) #тут разбило на предлежения
from nltk.corpus import stopwords # стоп слова
nltk_stopwords_eng = stopwords.words('english')
print(nltk_stopwords_eng)

from nltk.probability import FreqDist
word_frequencies = FreqDist(word_tokenize(text.lower()))
print(word_frequencies)
print(word_frequencies.items())

words = nltk.word_tokenize(text.lower())
print(f'Количество слов в тексте: {len(text.split())}')
print(f'Количество уникальных слов в тексте: {len(set(text.split()))}')
print(f'Количество токенов в тексте: {len(words)}')
print(f'Количество уникальных токенов в тексте: {len(set(words))}')


#слова которые встречаются больше чем сколько то раз
frequent_tokens = []
for token, frequency in word_frequencies.items():
    if frequency >= 30:
        frequent_tokens.append((token,frequency))
        print('Ура! Частое')
    else:
        print('Ура! Редкое')
print(frequent_tokens)

# слова которые встретились 1 раз
word_frequencies.hapaxes()



# облако слов

from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=200). generate(text)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show() 



