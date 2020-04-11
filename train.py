import math
import numpy as np
import os
import random
import tensorflow as tf
from matplotlib import pylab
from collections import Counter
import csv
import numpy as np
import tensorflow as tf
import codecs
from tensorflow.contrib.rnn import LSTMCell,LSTMStateTuple
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.python.layers.core import Dense
import logging
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import csv
import os.path

dropout=0
GO_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2
epochs=0
decode_method=1
uni_directional=0

input_filename="train.combined"
output_filename="summaries.txt" 
input_filename1="test.combined"
input_val="dev.combined"
output_val="summaries_val.txt"
input_test="test.combined"
#output_test="summaries_val.txt"
pred_out="pred_out.txt"
pred_out1="pred_out1.txt"

vocab_filename="vocabu.txt"
 

#load vocabulary
def load_vocab(filename):
    vocab = {}
    with open(filename) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx
    return vocab 
     

vocab = load_vocab(vocab_filename) 



params = {
        'vocab_size': len(vocab),
        'batch_size': 5,
        'input_max_length': 122,
        'output_max_length': 75,
        'embed_dim': 256,
        'num_units': 512
    }
vocab_size = params['vocab_size']
embed_dim = params['embed_dim']
num_units = params['num_units']
input_max_length = params['input_max_length']
output_max_length = params['output_max_length']
batch_size=params['batch_size']





 
def get_rev_vocab(vocab):
    return {idx: key for key, idx in vocab.items()}

def formatter(arr, vocab,outfile):
    rev_vocab = get_rev_vocab(vocab)
    with open(outfile,'a') as f:             
        for i in range(batch_size):
            res = []
            for j in range(len(arr[i])):
                res.append(rev_vocab[arr[i][j]])
            print(' '.join(res))
            f.write(' '.join(res))
            f.write('\n')
    f.close()
           
def newformatter(arr, vocab):
    rev_vocab = get_rev_vocab(vocab)
    res=[]           
    for j in range(len(arr)):
        res.append(rev_vocab[arr[j]])
    return res

def tokenize_and_map(line, vocab):
   return [vocab.get(token, UNK_TOKEN) for token in line.lower().split(' ')]


def make_input_fn(
    input_filename, output_filename, vocab,
    input_max_length, output_max_length,
    input_process=tokenize_and_map, output_process=tokenize_and_map):

    inpu=[]
    out=[]
    with open(input_filename) as finput:
        with open(output_filename) as foutput:
            for in_line in finput:
                out_line = foutput.readline()
                temp1=input_process(in_line, vocab)[:input_max_length - 1] + [END_TOKEN]
                inpu.append(temp1)
                temp2=output_process(out_line, vocab)[:output_max_length - 1] + [END_TOKEN]
                out.append(temp2)
    return inpu,out

inpu_tr,out_tr=make_input_fn(
        input_filename,
        output_filename,
        vocab, params['input_max_length'], params['output_max_length'])

inpu_val,out_val=make_input_fn(
        input_val,
        output_val,
        vocab, params['input_max_length'], params['output_max_length'])


def make_input_batch(in_matrix,out_matrix,start_index,batch_size,input_max_length,output_max_length):
    inputs, outputs = [], []
    input_length, output_length = 0, 0
   

    for i in range(batch_size):
        inputs.append(in_matrix[start_index+i])
        outputs.append(out_matrix[start_index+i])
        input_length = max(input_length, len(inputs[-1]))
        output_length = max(output_length, len(outputs[-1]))
        # Padding on right with </S> token.
    for i in range(batch_size):
        inputs[i] += [END_TOKEN] * (input_length - len(inputs[i]))
        outputs[i] += [END_TOKEN] * (output_length - len(outputs[i]))

    return inputs,outputs



def make_input_fn_test(
    input_filename, vocab,
    input_max_length,
    input_process=tokenize_and_map):

  inpu=[]
  with open(input_filename) as finput:
    for in_line in finput:
      temp1=input_process(in_line, vocab)[:input_max_length - 1] + [END_TOKEN]
      inpu.append(temp1)   
  return inpu
  
inpu_test=make_input_fn_test(
        input_test,
        vocab, params['input_max_length'])

def make_input_batch_test(in_matrix,start_index,batch_size,input_max_length):
  inputs = []
  input_length = 0
  for i in range(batch_size):
      inputs.append(in_matrix[start_index+i])
      input_length = max(input_length, len(inputs[-1]))
        # Pad me right with </S> token.
  for i in range(batch_size):
      inputs[i] += [END_TOKEN] * (input_length - len(inputs[i]))
        
  return inputs




#Model for train
def TrainModel(input1, output1):
    batch_size = tf.shape(input1)[0]
    start_tokens = tf.zeros([batch_size], dtype=tf.int64)
    train_output = tf.concat([tf.expand_dims(start_tokens, 1), output1], 1)
    input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(input1, 1)), 1)
    output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_output, 1)), 1)
    input_embed = layers.embed_sequence(
        input1, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed')
    output_embed = layers.embed_sequence(
        train_output, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed', reuse=True)
    with tf.variable_scope('embed', reuse=True):
        embeddings = tf.get_variable('embeddings')
    if(dropout==1):
        cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units),dropout_prob)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)#initial_state = cell.zero_state([batch_size], dtype=tf.float32)
    if(uni_directional==1):
      encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell, input_embed, dtype=tf.float32)
      num_units1=512
    else:
      ((encoder_fw_outputs,encoder_bw_outputs), (encoder_fw_final_state,encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,cell_bw=cell,inputs=input_embed,dtype=tf.float32, 
      time_major=True)
      encoder_outputs=tf.concat((encoder_fw_outputs,encoder_bw_outputs),2)
      encoder_final_state_c=tf.concat((encoder_fw_final_state.c,encoder_bw_final_state.c),1)
      encoder_final_state_h=tf.concat((encoder_fw_final_state.h,encoder_bw_final_state.h),1)
      encoder_final_state=LSTMStateTuple(c=encoder_final_state_c,h=encoder_final_state_h)
      num_units1=1024
    train_helper = tf.contrib.seq2seq.TrainingHelper(output_embed, output_lengths)
    pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens=tf.to_int32(start_tokens), end_token=1)
    decoder_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units1)
    projection_layer = Dense(units=vocab_size,use_bias=True)
    def decode(helper, scope, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
          if(decode_method==1):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=num_units1, memory=encoder_outputs,
                memory_sequence_length=input_lengths)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, attention_mechanism,alignment_history=True,attention_layer_size=num_units1/2)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attn_cell, vocab_size, reuse=reuse
            )
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=out_cell, helper=helper,
                initial_state=out_cell.zero_state(
                    dtype=tf.float32, batch_size=batch_size),output_layer=projection_layer)
          else:
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                decoder_cell, vocab_size, reuse=reuse)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=out_cell, helper=helper,
                initial_state=out_cell.zero_state(
                    dtype=tf.float32, batch_size=batch_size),output_layer=projection_layer)
          outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=output_max_length
            )
        return outputs[0],outputs[1]
    
    if(epochs<5):
      [train_outputs,decoder_state] = decode(train_helper, 'decode')
    else:
      [train_outputs,decoder_state] = decode(pred_helper, 'decode', reuse=True)
    weights = tf.to_float(tf.not_equal(train_output[:, :-1], 1))
    loss = tf.contrib.seq2seq.sequence_loss(
        train_outputs.rnn_output, output1, weights=weights)
    train_op = layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer=params.get('optimizer', 'Adam'),
        learning_rate=params.get('learning_rate', lr),
        summaries=['loss', 'learning_rate'])
    return loss,train_op,train_outputs,decoder_state
  
def ValModel(input1, output1):
    batch_size = tf.shape(input1)[0]
    start_tokens = tf.zeros([batch_size], dtype=tf.int64)
    train_output = tf.concat([tf.expand_dims(start_tokens, 1), output1], 1)
    input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(input1, 1)), 1)
    output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_output, 1)), 1)
    input_embed = layers.embed_sequence(
        input1, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed')
    output_embed = layers.embed_sequence(
        train_output, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed', reuse=True)
    with tf.variable_scope('embed', reuse=True):
        embeddings = tf.get_variable('embeddings')
    if(dropout==1):
        cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units),dropout_prob)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)#initial_state = cell.zero_state([batch_size], dtype=tf.float32)
    if(uni_directional==1):
      encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell, input_embed, dtype=tf.float32)
      num_units1=512
    else:
      ((encoder_fw_outputs,encoder_bw_outputs), (encoder_fw_final_state,encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,cell_bw=cell,inputs=input_embed,dtype=tf.float32, 
      time_major=True)
      encoder_outputs=tf.concat((encoder_fw_outputs,encoder_bw_outputs),2)
      encoder_final_state_c=tf.concat((encoder_fw_final_state.c,encoder_bw_final_state.c),1)
      encoder_final_state_h=tf.concat((encoder_fw_final_state.h,encoder_bw_final_state.h),1)
      encoder_final_state=LSTMStateTuple(c=encoder_final_state_c,h=encoder_final_state_h)
      num_units1=1024
    train_helper = tf.contrib.seq2seq.TrainingHelper(output_embed, output_lengths)
    pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens=tf.to_int32(start_tokens), end_token=1)
    decoder_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units1)
    projection_layer = Dense(units=vocab_size,use_bias=True)
    def decode(helper, scope, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
          if(decode_method==1):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=num_units1, memory=encoder_outputs,
                memory_sequence_length=input_lengths)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, attention_mechanism, attention_layer_size=num_units1/2)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attn_cell, vocab_size, reuse=reuse
            )
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=out_cell, helper=helper,
                initial_state=out_cell.zero_state(
                    dtype=tf.float32, batch_size=batch_size),output_layer=projection_layer)
          else:
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                decoder_cell, vocab_size, reuse=reuse)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=out_cell, helper=helper,
                initial_state=out_cell.zero_state(
                    dtype=tf.float32, batch_size=batch_size),output_layer=projection_layer)
          outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=output_max_length
            )
        return outputs[0]
    
    if(epochs<5):
      train_outputs = decode(train_helper, 'decode')
    else:
      train_outputs = decode(pred_helper, 'decode', reuse=True)
    weights = tf.to_float(tf.not_equal(train_output[:, :-1], 1))
    loss = tf.contrib.seq2seq.sequence_loss(
        train_outputs.rnn_output, output1, weights=weights)
    return loss,train_outputs
  
def TestModel(input1):
    batch_size = tf.shape(input1)[0]
    start_tokens = tf.zeros([batch_size], dtype=tf.int64)
    input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(input1, 1)), 1)
    input_embed = layers.embed_sequence(
        input1, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed')
    with tf.variable_scope('embed', reuse=True):
        embeddings = tf.get_variable('embeddings')
    if(dropout==1):
        cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units),1)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)#initial_state = cell.zero_state([batch_size], dtype=tf.float32)
     if(uni_directional==1):
      encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell, input_embed, dtype=tf.float32)
      num_units1=512
    else:
      ((encoder_fw_outputs,encoder_bw_outputs), (encoder_fw_final_state,encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,cell_bw=cell,inputs=input_embed,dtype=tf.float32, 
      time_major=True)
      encoder_outputs=tf.concat((encoder_fw_outputs,encoder_bw_outputs),2)
      encoder_final_state_c=tf.concat((encoder_fw_final_state.c,encoder_bw_final_state.c),1)
      encoder_final_state_h=tf.concat((encoder_fw_final_state.h,encoder_bw_final_state.h),1)
      encoder_final_state=LSTMStateTuple(c=encoder_final_state_c,h=encoder_final_state_h)
      num_units1=1024
    pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens=tf.to_int32(start_tokens), end_token=1)
    decoder_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units1)
    projection_layer = Dense(units=vocab_size,use_bias=True)
    def decode(helper, scope, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
          if(decode_method==1):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=num_units1, memory=encoder_outputs,
                memory_sequence_length=input_lengths)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, attention_mechanism, attention_layer_size=num_units1/2)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attn_cell, vocab_size, reuse=reuse
            )
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=out_cell, helper=helper,
                initial_state=out_cell.zero_state(
                    dtype=tf.float32, batch_size=batch_size),output_layer=projection_layer)
          else:
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                decoder_cell, vocab_size, reuse=reuse)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=out_cell, helper=helper,
                initial_state=out_cell.zero_state(
                    dtype=tf.float32, batch_size=batch_size),output_layer=projection_layer)
          outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=output_max_length
            )
        return outputs[0]
    train_outputs = decode(pred_helper, 'decode', reuse=True)
    return train_outputs
 
#Calling Model 
parser=argparse.ArgumentParser()
parser.add_argument('--lr', type=float)		
parser.add_argument('--batch_size', type=float)	
parser.add_argument('--dropout_prob', type=str)
parser.add_argument('--decode_method', type=str)
args = parser.parse_args()

lr=args.lr
batch_size=args.batch_size
dropout_prob=args.dropout_prob
decode_method=args.decode_method



    
with tf.variable_scope('root'):
  inp1 = tf.placeholder(tf.int64, shape=[None, None])
  output1 = tf.placeholder(tf.int64, shape=[None, None])
  train_op, loss,train_outputs,decoder_state = TrainModel(inp1,output1)
  initializer = tf.global_variables_initializer()

with tf.variable_scope('root', reuse=True):
  inp2 = tf.placeholder(tf.int64, shape=[None, None])
  output2 = tf.placeholder(tf.int64, shape=[None, None])
  val_loss,val_out = ValModel(inp2,output2)
  
  
with tf.variable_scope('root', reuse=True):
  inp3 = tf.placeholder(tf.int64, shape=[None, None])
  test_out = TestModel(inp3)  

sess = tf.Session()

sess.run(initializer)
tr_num_batches=500
val_num_batches=2
loss_tr=0
loss_val=0
prev_loss=0
patience_count=0
patience=5

for steps in range(10*tr_num_batches):
  inputs,outputs = make_input_batch(
        inpu_tr,
        out_tr,(batch_size),
        batch_size, params['input_max_length'], params['output_max_length'])
  _,loss1,train_out,outnew=sess.run([train_op, loss,train_outputs,decoder_state.alignment_history.stack()],feed_dict={inp1:inputs,output1:outputs})
  loss_tr=loss_tr+loss1
  if(steps!=0 and (steps+1)%tr_num_batches==0):
    loss_tr=loss_tr/tr_num_batches
    epochs=epochs+1
    print("Training Loss: %i: %f" %(epochs,loss_tr))
    loss_tr=0
    for steps_val in range(val_num_batches):
      #loss2=sess.eval([valloss], feed_dict={inp2: inputs2,output2:outputs2})
      inputs2,outputs2 = make_input_batch(
        inpu_val,
        out_val,steps_val*batch_size,
        batch_size, params['input_max_length'], params['output_max_length'])
      loss2,val_out2=sess.run([val_loss,val_out],feed_dict={inp2: inputs2,output2:outputs2})
      loss_val=loss_val+loss2
    loss_val=loss_val/val_num_batches
    print("Validation Loss: %i: %f" %(epochs,loss_val))
    if(prev_loss>0):
      if(prev_loss-loss_val>0):
        patience_count=0
      else:
        patience_count=patience_count+1
    if patience_count>patience or epochs==5:
      for steps_val in range(val_num_batches):
        inputs2,outputs2 = make_input_batch(inpu_val,out_val,steps_val*batch_size,
        batch_size, params['input_max_length'], params['output_max_length'])
        loss2,val_out2=sess.run([val_loss,val_out],feed_dict={inp2: inputs2,output2:outputs2})
        formatter(val_out2.sample_id, vocab,pred_out)
      print("early stopping")
      break
    prev_loss=loss_val
    
for x in range(20):
    inputs3 = make_input_batch_test(inpu_test,x*batch_size,batch_size, params['input_max_length'])
    test_out1=sess.run(test_out,feed_dict={inp3: inputs3})
    formatter(test_out1.sample_id,vocab,pred_out1)
    
    
  if epochs==10:
    #Attention visulalization for k in range (0,50) 
    k=25
    x=outnew.reshape(batch_size,output_max_length,input_max_length)      
    activation_map = np.squeeze(x[0])[0:output_max_length, 0:input_max_length]   
    plt.clf()
    f = plt.figure(figsize=(8, 8.5))
    ax = f.add_subplot(1, 1, 1)
    i = ax.imshow(activation_map, interpolation='nearest', cmap='gray')
    cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
    cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xlabel('Probability', labelpad=2)    
    ax.set_yticks(range(89))
    y1=newformatter(train_out.sample_id[k], vocab)
    ax.set_yticklabels(newformatter(train_out.sample_id[0], vocab))
    ax.set_xticks(range(122))
    x1=newformatter(inputs[k], vocab)
    ax.set_xlabel('Input Sequence')
    ax.set_xticklabels(newformatter(inputs[0], vocab))
    ax.set_ylabel('Output Sequence')
    ax.grid()
    
    
    