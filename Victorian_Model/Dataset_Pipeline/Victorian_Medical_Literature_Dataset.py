
from torch.utils.data import Dataset
import json

class Old_Medical_Dataset(Dataset):

    def __init__(self,clean_text_function,txt_path):
        self.txt_path=txt_path
        self.clean=clean_text_function
    def __len__(self):
        return len(self.txt_path)
    
    def __getitem__(self,index):
        with open(self.txt_path,"r") as file:
            json_file_contents=json.load(file)
            samples=(json_file_contents['all_samples'])
        (sample,) = samples[index].values()
        clean_sample=self.clean(sample)
        clean_label=self.clean(sample)
        return(clean_sample,clean_label)
    

