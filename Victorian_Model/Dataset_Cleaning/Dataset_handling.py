import os
from collections import Counter
import re
import random
import json

def clean_file(file_path):
    with open(file_path,"r",encoding="utf-8") as dirty_file:
        dirty_file_contents=dirty_file.read()
        lowercase_dirty_file_contents=dirty_file_contents.lower()
        single_line=lowercase_dirty_file_contents.replace("\n"," ")
        single_line_one_space=single_line.replace("  "," ")
        single_line_one_space_no_punctuation=re.sub(r"[?,!'&:;\{\[\(\)\]\}\"°]","",single_line_one_space)
        
        text = re.sub(r"\\u[0-9a-fA-F]{4}", "", single_line_one_space_no_punctuation)

        # Step 2: Remove actual Unicode characters by range if needed
        # Example: Remove Japanese hiragana and katakana ranges
        text = re.sub(r"[\u0080-\uFFFF]", "", text)  # Adjust range as needed

        # Additional step: Remove all control characters (optional)
        text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)
    return(text)

dirty_file_folder=r"C:\Users\James\Documents\Memory\Dirty Text Files"
dirty_files=(os.listdir(dirty_file_folder))
all_subtexts=[]
for dirty_file in dirty_files:
    dirty_file_path=os.path.join(dirty_file_folder,dirty_file)
    if dirty_file_path == r"C:\Users\James\Documents\Memory\Dirty Text Files\Medical Terms":
        pass
    else:  
        print(dirty_file)
        purified=clean_file(dirty_file_path)
        unique_words=purified.split(" ")

        full_stop_positions = []
        start = 0
        while True:
            pos = purified.find(".", start)
            if pos==-1:
                break
            full_stop_positions.append(pos)
            start = pos + 1

        number_of_full_spots=(len(full_stop_positions))
        generate_texts=True
        subtexts=[]
        position=0
        sequences=1
        while generate_texts==True:
            index_a=position
            index_b=position+(random.choice([2,3,4]))
            start=full_stop_positions[index_a]
            if index_b<number_of_full_spots:
                end=full_stop_positions[index_b]
                if index_a==0:
                    sub_text=purified[0:end]

                else:
                    sub_text=purified[start:end]
            else:
                sub_text=purified[start:number_of_full_spots]
                generate_texts=False
            if len(sub_text)>20:
                sample=f"{dirty_file} Sample {sequences}"
                subtexts.append({sample:sub_text})
                sequences+=1
            position=index_b
        all_subtexts=all_subtexts+subtexts
        unique_words=Counter(unique_words)
all_samples={"all_samples":all_subtexts}
print(len(all_subtexts))

maximum_sequence_size=0
for each_dict in all_subtexts:
    number_of_characters=len(list(each_dict.values())[0])
    if number_of_characters> maximum_sequence_size:
        maximum_sequence_size=number_of_characters

print(maximum_sequence_size)
with open("medical_dataset_written","w",) as medical_json_file:
    json.dump(all_samples,medical_json_file,indent=1)
