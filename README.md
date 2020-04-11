# Structured-Data-to-Text-Descriptions-ses2seq-model

The objective is to generate a description suited for the given table using sequence to sequence network given a table of facts.

Configuration:

Using tensorflow, training a sequence-to-sequence model using the encoder-decoder archi-
tecture. The overall structure of the network is as follows:

(a) INEMBED: A feedforward layer of size |Vs| × inembsize, where Vs is the source
vocabulary (i.e. set of words) and inembsize is the output size of the layer. Use
inembsize = 256

(b) ENCODER:

– BASIC ENCODER: Bidirectional LSTM layer with encsize outputs in either direction. encsize = 512

– HIERARCHICAL ENCODER : The hierarchical encoder will fist encode the tokens using a token level LSTM. Note
that the token level bidirectional LSTM should be shared across all the rows. The final state of this LSTM should be then concatenated with the ‘field’ word embedding and then should be passed through another bidirectional LSTM which basically runs over the first column of the given table. For instance, temperature is a field, with token string time 17-30 min 33
max 60 mean 44 .... GT is the bidirectional LSTM over these token string which will output the final hidden state HT which essentially encodes the information about the temperature field. This state will be then concatenated
with Wemb(temperature). These representations for each field will then be passed through another bidirectional LSTM GF . Pass the final hidden state of this encoder to generate the text descriptions.

(c) ATTENTION: Incorporated an attention mechanism over the Basic Encoder consisting of: 

(i) a single feedforward network whose inputs are the previous decoder state, previous decoder output’s embedding and
the annotation vector (encoder output) under consideration. It outputs a single
attention score. 

(ii) a softmax layer over the attention scores from each annotation
vector to convert it to a probability value. The attention weights from the softmax
layer are used to linearly interpolate the annotation vectors. The resulting context
vector will be an input to the decoder along with the decoder output in the
previous timestep.

(d) DECODER: LSTM layer with decsize outputs. decsize = 512

(e) SOFTMAX: softmax layer for classification: decsize inputs and Vt outputs, where Vt is the target language vocabulary

(f) OUTEMBED: A feedforward layer of size |Vt| × outembsize, where embsize
is the output size of the layer. Use outembsize = 256. This layer is used for
obtaining the embedding of the output character and feeding it to the input of
the decoder for the next time step

