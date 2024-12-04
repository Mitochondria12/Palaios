from Victorian_Model.Training_Modules.Model_Training import create_dataloader,padding_batch,training_batch,padding_batch_labels_optimized
from Victorian_Model.Model_Architecture.Attention_Model import Transformer
from Victorian_Model.Dataset_Pipeline.Pretrained_Tokenisation import text_token_convertor
import torch.nn as nn
import torch 
import argparse
samples=10000

def command_line_parameters():
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch_size",required=True,type=int, help="the batch size you want to run the model with.")
    output=parser.parse_args()
    return(output)

if __name__ == "__main__":
    command=command_line_parameters()
    batch_size=(command.batch_size)
    model=Transformer(12)
    model.train()
    #assigning the loss formula
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    #assigning the optimiser which calculates the gradients and performs the update methods
    optimiser=torch.optim.SGD(model.parameters(),lr=0.3)
    number_of_batches=round(samples/batch_size)
    data_holder=create_dataloader(batch_size)
    
    for batch in range(1,number_of_batches+1):
        print(f"Batch {batch} loaded.")
        train_features, train_labels = next(iter(data_holder))
        padded_batch_features=padding_batch(train_features)
        padded_batch_labels=text_token_convertor(train_labels)

        training_batch(model,optimiser,loss_fn,padded_batch_features,padded_batch_labels)
       
