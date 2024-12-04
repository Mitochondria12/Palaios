def label_zero_shot(token_id):
    #create a vector tensor
    current_token=torch.zeros([30522])
    #modify the tensor value at the position of the token
    current_token[token_id]=1
    return(current_token)

def padding_batch_labels(labels):
    padded_tokens=text_token_convertor(labels)

    batch_sequences_zero=[]
    for sequence in padded_tokens["input_ids"]:
        sequence_zero=[]
        for token in sequence:
            sequence_zero.append(label_zero_shot(token))
        batch_sequences_zero.append(sequence_zero)
    return(batch_sequences_zero)
