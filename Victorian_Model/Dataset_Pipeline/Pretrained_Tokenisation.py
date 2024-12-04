from transformers import AutoTokenizer
import torch

"""This function takes text as input, puts it into a bert tokeniser to get text into a sequence of tokens. If input is a batch of text, then the sequence
of each text is the same as the longest text number of tokens. The tokeniser is a dictionary lookup function, where each word is lookedup for its corresponding 
numerical representation. """
def text_token_convertor(text):
    tokeniser = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokeniser_dictionary = tokeniser(text,padding="longest")
    token_ids = torch.tensor(tokeniser_dictionary["input_ids"]) 

    return(token_ids)

