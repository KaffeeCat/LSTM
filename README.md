本贴使用MXNet形式的伪代码来展示LSTM的使用，重点在陈述如何将LSTM作为工具，应用在实际需求中。

这里有一篇很好的帖子来讲述LSTM，[在这篇帖子里](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) ，LSTM网络如下图所示：
![LSTM-image](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

第一步：定义LSTM的State，每一个LSTM的输入为当前状态x，以及前一个LSTM cell的输出状态[c, h]：
``` Python
immutable LSTMState
  c :: mx.SymbolicNode # information flow (just like a conveyor belt)
  h :: mx.SymbolicNode # output value of previous LSTM cell
end
``` 

第二步：定义LSTM中的权重（shared at every time）：
``` Python
immutable LSTMParam
  i2h_W :: mx.SymbolicNode # 输入x相关的weights
  i2h_b :: mx.SymbolicNode # 输入x相关的bias
  h2h_W :: mx.SymbolicNode # 从上一个cell传入的h相关的weights
  h2h_b :: mx.SymbolicNode # 从上一个cell传入的h相关的bias
end
``` 
第三步：定义LSTM cell
``` Python
fuction lstm_cell(  input_data::SymbolicNode, # input x
                    prev_state::LSTMParam, # previous state
                    num_hidden::Int = 512)
                    
   # LSTM cell的输入为当前状态x, 以及上一个LSTM cell的状态输出[c, h]，我们对输入的x及h建立全连接网络，训练学习其中的weights和bias
   i2h = mx.FullyConnected(data=input_data, weight=param.i2h_W, bias=param.i2h_b, num_hidden=4*num_hidden)
   h2h = mx.FullyConnected(data=prev_state, weight=param.h2h_W, bias=param.h2h_b, num_hidden=4*num_hidden)
   
   # LSTM的gates，其输入为FullyConnected网络i2h和h2h的输出，将其切成四份，作为LSTM cell中4个gate的输入
   gates = mx.SliceChannel(i2h + h2h, num_outputs=4)
   
   # LSTM中每个gate的运算，参照Colah的博客：http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 
   in_gate = mx.Activation(gates[1], act_type=:sigmoid)
   in_trans = mx.Activation(gates[2], act_type=:tanh)
   forget_gate = mx.Activation(gates[3], act_type=:sigmoid)
   out_gate = mx.Activation(gates[4], act_type=:sigmoid)
   
   # LSTM的gates输出，根据公式，计算出输出到下一个LSTM cell的[c, h]
   next_c = (forget_gate .* prev_state.c) + (in_gate .* in_trans)
   next_h = out_gate .* mx.Activation(next_c, act_type=:tanh)
   
   return LSTMState(next_c, next_h)
``` 
第四步：完整（展开的）LSTM.
``` Python
fuction LSTM( n_layer::Int, # 多少层Stack LSTM layers
              seq_len::Int, # 多少时间节点
              dim_hidden::Int,  # LSTM cell中i2h和h2h连接每个gate的网络隐藏层节点数
              dim_embed::Int, # 输入data数据编码用的全连接网络的隐藏层节点数
              n_class::Int) # 输出的类别总数
   
   # 对所有weights和states进行定义
   embed_W = mx.Variable(symbol(name, "_embed_weight")) # 对输入x编码的weights.
   pred_W = mx.Variable(symbol(name, "_pred_weight")) # 在每一个LSTM cell中做预测判断使用的weights
   pred_b = mx.Variable(symbol(name, "_pred_bias")) #  在每一个LSTM cell中做预测判断使用的bias
   
   # 对每一个LSTM cell进行定义
   layer_param_states = map(1:n_layer) do i
      param = LSTMParam(  mx.Variable(symbol(name, "_l$(i)_i2h_weight")),
                          mx.Variable(symbol(name, "_l$(i)_i2h_bias")),
                          mx.Variable(symbol(name, "_l$(i)_h2h_weight")),
                          mx.Variable(symbol(name, "_l$(i)_h2h_bias")))
      state = LSTMState(  mx.Variable(symbol(name, "_l$(i)_init_c")),
                          mx.Variable(symbol(name, "_l$(i)_init_h")))
      (param, state)
   end
   
   outputs = mx.SymbolicNode[]  # 用来存放所有LSTM cell的输出结果
   
   for t = 1:seq_len
      data = mx.Variable(symbol(name, "_data_$t"))
      label = mx.Variable(symbol(name, "_label_$t"))
      hidden = mx.FullyConnected(data=data, weight=embed_W, num_hidden=dim_embed, no_bias=true)
      
      # Stack LSTM cells
      for i = 1:n_layer
          l_param, l_state = layer_param_states[i]
          next_state = lstm_cell(hidden, l_state, l_param, num_hidden=dim_hidden)
          hidden = next_state.h
          layer_param_states[i] = (l_param, next_state)
       end
       
       # 使用Softmax预测输出
       pred = mx.FullyConnected(data=hidden, weight=pred_W, bias=pred_b, num_hidden=n_class)
       smax = mx.SoftmaxOutput(pred, label)
       push!(outputs, smax)
   end
   
   # 在每一个时间节点上，LSTM cell的预测输出连接这Softmax操作，根据给予的样本label的进行后向反馈(back propagate)，LSTM state在时间序列上前后顺序相连，让系统根据时间序列后向反馈(back propagate)。然后，在时间序列的尾端，最后一个LSTM state后续再无连接，所以在下面，使用BlockGrad让最后一个state与所有的states相连，让其后向反馈0梯度，使整个计算图完整。
   for i = 1:n_layer
      l_param, l_state = layer_param_states[i]
      final_state = LSTMState(mx.BlockGrad(l_state.c),
                              mx.BlockGrad(l_state.h)) # BlockGrad operator back propagates 0-gradient
      layer_param_states[i] = (l_param, final_state)
   end
   
   return mx.Group(outputs...)
``` 
