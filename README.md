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

第二部：定义LSTM中的权重（shared at every time）：
``` Python
immutable LSTMParam
  i2h_W :: mx.SymbolicNode # 输入x相关的weights
  i2h_b :: mx.SymbolicNode # 输入x相关的bias
  h2h_W :: mx.SymbolicNode # 从上一个cell传入的h相关的weights
  h2h_b :: mx.SymbolicNode # 从上一个cell传入的h相关的bias
end
``` 
第三部：定义LSTM cell
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
