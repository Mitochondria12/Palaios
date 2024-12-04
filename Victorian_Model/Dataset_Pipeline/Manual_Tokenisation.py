#Here we will tokenise the volcabulary
import json
from collections import Counter
import string
from transformers import AutoTokenizer


file_path=r"C:\Users\James\Documents\Memory\medical_dataset_written"
with open(file_path,"r") as file:
    json_file_contents=json.load(file)

samples=(json_file_contents['all_samples'])
(value,) = samples[5].values()
print(value)
all_text=""
for dict in samples:
    text=list(dict.values())[0]
    
    all_text+=text
words = all_text.translate(str.maketrans('', '', string.punctuation))
words=(words.split())

word_freq = Counter(words)




