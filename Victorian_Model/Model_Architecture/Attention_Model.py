import torch.nn as nn
import torch
import torch.nn.functional as functional          
import torch.optim.sgd
import transformers
from transformers import AutoTokenizer,AutoModel

#attention block
class Multi_Self_Attention_Head_Block(nn.Module):

    def __init__(self,embedding_dimensions,heads):
        super(Multi_Self_Attention_Head_Block,self).__init__()
        self.heads=heads
        self.richness=embedding_dimensions

        self.query=nn.Linear(self.richness,self.richness*self.heads,dtype=torch.float32)
        self.key=nn.Linear(self.richness,self.richness*self.heads,dtype=torch.float32)
        self.value=nn.Linear(self.richness,self.richness*self.heads,dtype=torch.float32)

        self.final_projection=nn.Linear(self.heads*self.richness,self.richness,dtype=torch.float32)

    def forward(self,input):
        
        #shape(batch_size,sequence_length,embedding_size)
        input_shape=input.shape
        
        #shape(batch_size,head_size,sequence_length,embedding_size)
        query_output=torch.reshape(self.query(input),(input_shape[0],input_shape[1],self.heads,self.richness)).transpose(1, 2)
        key_output=torch.reshape(self.key(input),(input_shape[0],input_shape[1],self.heads,self.richness)).transpose(1, 2)
        value_output=torch.reshape(self.value(input),(input_shape[0],input_shape[1],self.heads,self.richness)).transpose(1, 2)

        #shape(batch_size,head_size,sequence_length,sequence_length)
        similarity_scores=torch.matmul(query_output,key_output.transpose(-2,-1))
        mask=torch.tril(similarity_scores,0)

        similarity_scores=torch.where(mask==0,-99999,similarity_scores)
        
        similarity_scores=functional.softmax(similarity_scores,dim=-1,dtype=torch.float32)  

        output_tensor_quad=torch.matmul(similarity_scores,value_output)

        output_tensor_quad=torch.transpose(output_tensor_quad,-3,-2)

        output_tensor_quad=torch.reshape(output_tensor_quad,((input_shape[0],input_shape[1],self.heads*self.richness)))

        multi_head_attention=self.final_projection(output_tensor_quad)
        
        return(multi_head_attention)
    
class Normalisation_Block(nn.Module):

    def __init__(self):
        super(Normalisation_Block,self).__init__()
        self.layer=nn.Linear(768,768)

    def forward(self,input):
        shape=input.shape
        sequence=torch.flatten(input)
        mean=sequence.mean()
        normalised_sequence=sequence-torch.tensor([mean for element in range(sequence.size(-1))])
        normalised_input=torch.reshape(normalised_sequence,(shape))
        return(normalised_input)
    
class Previous_Input(nn.Module):

    def __init__(self):
        super(Previous_Input,self).__init__()

    def forward(self,input):

        output=input[0]+input[1]

        return(output)

class Feed_Forward_Network(nn.Module):

    def __init__(self):
        super(Feed_Forward_Network,self).__init__()
        self.weights=nn.Linear(768,768)
        self.leaky_relu=nn.LeakyReLU()
    def forward(self,input):

        output=self.leaky_relu(self.weights(input))

        return(output)

class Transformer_Layer(nn.Module):

    def __init__(self):
        super(Transformer_Layer,self).__init__()
        self.attn_module=Multi_Self_Attention_Head_Block(768,3)
        self.norm_module=Normalisation_Block()
        self.addin_module=Previous_Input()
        self.feed_module=Feed_Forward_Network()

    def forward(self,input):
        attention_module=self.attn_module(input)
        normalisation_module=self.norm_module(attention_module)
        addition_module= self.addin_module([normalisation_module,input])
        feed_forward=self.feed_module(addition_module)
        return(feed_forward)


class Transformer(nn.Module):

    def __init__(self,depth_level):
        super(Transformer, self).__init__()
        # Define the sequence of Transformer layers
        self.Transformer_Block = nn.ModuleList([Transformer_Layer() for depth in range(depth_level)])
        self.logits_module=nn.Linear(768,30522)
        self.softmax=nn.LogSoftmax(dim=-1)
        self.depth_level=depth_level

    def forward(self, input):
        # Run input through the Transformer_Block sequence
        for layer in self.Transformer_Block:
            output = layer(input)
            input=output
        #This is automatically implemented in cross cateogorical function.
        #output=self.softmax(self.logits_module(output))
        output=(self.logits_module(output))
        return output
    


