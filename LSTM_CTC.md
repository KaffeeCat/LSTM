本贴参考[MXNet官方文档](http://mxnet.io/tutorials/speech_recognition/baidu_warp_ctc.html)中LSTM+CTC的例子，对该例子进行分析和修整。

Example介绍：<br>
1. 输入：80x30像素大小的图片，内含4个阿拉伯数字。<br>
2. 目的：将该图像每一列（30 dim vector）作为LSTM的输入，使用LSTM+CTC进行结果预测<br>

``` Python
def lstm_unroll(  num_lstm_layer, 
                  seq_len, 
                  num_hidden, 
                  num_label
                  ):
                  
    # ================================================
    # 创建Stack LSTM
    param_cells = []  # 存放每个LSTM cell的LSTMParam
    last_states = []  # 存放每个LSTM cell的LSTMState
    
    for i in range(num_lstm_layer):
        param_cells.append( LSTMParam(i2h_weight  = mx.sym.Variable("l%d_i2h_weight" % i),
                                      i2h_bias    = mx.sym.Variable("l%d_i2h_bias" % i),
                                      h2h_weight  = mx.sym.Variable("l%d_h2h_weight" % i),
                                      h2h_bias    = mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c = mx.sym.Variable("l%d_init_c" % i),
                          h = mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    
    assert(len(last_states) == num_lstm_layer)
    
    # ================================================
    # 将80x30大小图像的每一列作为LSTM输入，总共有80个输入
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    
    wordvec = mx.sym.SliceChannel(data=data, num_outputs=seq_len, squeeze_axis=1) # 将图像纵向切分成seq_len=80份
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]
        for 1 in range(num_lstm_layer):
            next_state = lstm(  num_hidden, 
                                indata      = hidden, 
                                prev_state  = last_states[i], 
                                param       = param_cells[i], 
                                seqidx      = seqidx, 
                                layeridx    = 1)
            hidden = next_state.h
            last_states[i] = next_state
        
        hidden_all.append(hidden)
    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=11)
    
    label = mx.sym.Reshape(data=label, target_shape=(0,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm = mx.sym.WarpCTC(data=pred, label)
    
``` 
