from transformers import AutoModel
import torch
"""This function takes a batch of numerical sequences, feeds it into a BERT LLM model which then gives an embedding output
for each sequence. This is the vector which is fed into the BERT classification model used to predict the token and contains
rich dimensional token representation. The token has to be reshaped so that it outputs shape(sequence_length,embedding_dimension) as it output is
shape(batch,sequence_length,embedding_dimension)."""
def text_embeddings(token_sequence):
        model = AutoModel.from_pretrained("bert-base-uncased")
        token_sequence = token_sequence.unsqueeze(0)
        with torch.no_grad():
            outputs = model(token_sequence)
        embeddings = outputs.last_hidden_state
        reshaped_embedding=torch.reshape(embeddings,(embeddings.shape[-2],embeddings.shape[-1]))
        return(reshaped_embedding)