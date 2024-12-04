import torch
def infere(model,optimiser,loss_fn,batch_features,batch_labels):
    
    #the Input,Ground_Truth is a tuple which contains the known information and the characteristic to be predicted
    batch_predictions=model(batch_features)
    batch_predicted_tokens=torch.argmax(batch_predictions,dim=-1)
    return(batch_predicted_tokens[-1])