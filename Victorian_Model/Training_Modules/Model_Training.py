import torch
from Victorian_Model.Dataset_Pipeline.Victorian_Medical_Literature_Dataset import Old_Medical_Dataset
from torch.utils.data import DataLoader
from Victorian_Model.Training_Modules.Embedding_Generator import text_embeddings
from Victorian_Model.Dataset_Pipeline.Pretrained_Tokenisation import text_token_convertor
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

#dataset comprises training features and ground truth
def training_batch(model,optimiser,loss_fn,batch_features,batch_labels):
    
    batch_predictions=model(batch_features)
    batch_predicted_tokens=batch_predictions[:,1:-1,:]
    batch_labels=batch_labels[:,2:]

    #Reformat model output and labels
    logits_reshaped = batch_predicted_tokens.reshape(-1, batch_predicted_tokens.size(-1))  
    targets_reshaped = batch_labels.reshape(-1)

    #Calculate loss
    loss=loss_fn(logits_reshaped,targets_reshaped)
    print("Model Loss for batch.")
    print(loss.item())

    #performs the backward propagation pass to calculate all the gradient changes to reduce the loss value
    loss.backward()

    #updates all the parameters of the models with the gradient change to reduce the loss value
    optimiser.step()

    #clears the loss backward gradient values so it doesn't interfere with the next calculation.
    optimiser.zero_grad()

    predicted_tokens=torch.argmax(batch_predicted_tokens,dim=-1)
    print(predicted_tokens)

    labels=torch.argmax(batch_labels,dim=-1)
    print(labels)

def clean_function(txt):
    return(txt)

"""This extracts the old medical literature dataset and puts it in a 
data storage object segregated into batches."""
def create_dataloader(batch_size):
    dataset_to_load=Old_Medical_Dataset(clean_function,r"C:\Users\James\Documents\GenAI\Palaios\Victorian_Model\Clean_Dataset\medical_dataset_written")
    train_dataloader = DataLoader(dataset_to_load, batch_size, shuffle=True)
    return(train_dataloader)

"""Function takes as input a text sequence processed by a function called text_token_convertor
to create a token representation of the sequence. Embeddings are generated from
the tokens which provide rich information. """
def tensor_generator(sequence):
    token_sequence=(text_token_convertor(sequence))
    #token sequence fed into a BERT LLM model to extract embedding representations for each token.
    embeddings=text_embeddings(token_sequence)
    return(embeddings)

"""This converts the features into a tensor sequence representation,
which are combined together to form a single tensor with the same
sequence size which is the largest sequence size of the batch."""
def padding_batch(batch):
    train_feature_tensors=list(map(tensor_generator,batch))
    padded_tensors=pad_sequence(train_feature_tensors,batch_first=True,padding_value=0)
    return(padded_tensors)

#chatgpt solution
def padding_batch_labels_optimized(labels, vocab_size=30522):
    # Convert text labels to tokenized sequences
    padded_tokens = text_token_convertor(labels)  # Assuming this is defined elsewhere

    padded_token_tensor=padded_tokens.convert_to_tensors("pt")["input_ids"]    
    
    # Determine the batch size and maximum sequence length
    batch_size = padded_token_tensor.shape[0]
    max_len = max(len(seq) for seq in padded_token_tensor)

    # Create a batch tensor for one-hot encoding
    batch_tensor = torch.zeros(batch_size, max_len, vocab_size)

    # Create indices for the one-hot positions
    batch_indices = torch.arange(batch_size).unsqueeze(1).repeat(1, max_len).flatten()
    seq_indices = torch.arange(max_len).repeat(batch_size)
    vocab_indices = padded_token_tensor.flatten()

    # Assign 1 to the corresponding positions
    batch_tensor[batch_indices, seq_indices, vocab_indices] = 1
    return batch_tensor

"""Input is a tensor matrix of batches and sequences, desired output is a 3d tensor with batch,sequence, dictionary as its dimensions. """