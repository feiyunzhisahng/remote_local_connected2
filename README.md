FCN_model

- batch_size 

  - ![image-20240506222924580](C:\Users\www14\AppData\Roaming\Typora\typora-user-images\image-20240506222924580.png)

  - ![image-20240506222939174](C:\Users\www14\AppData\Roaming\Typora\typora-user-images\image-20240506222939174.png)

  - ![image-20240506223923357](C:\Users\www14\AppData\Roaming\Typora\typora-user-images\image-20240506223923357.png)
  - ![image-20240506224506683](C:\Users\www14\AppData\Roaming\Typora\typora-user-images\image-20240506224506683.png)
  - ![image-20240506224515377](C:\Users\www14\AppData\Roaming\Typora\typora-user-images\image-20240506224515377.png)
  - 模型就一层太简单了，感觉batch_size还影响不到

- FCN_basic(1层)
  - ![image-20240507185132238](C:\Users\www14\AppData\Roaming\Typora\typora-user-images\image-20240507185132238.png)，这个等了很久
  - **待学习：怎么样充分利用gpu加速计算**
  - 设置了start_time和excution_time看训练时间，我将cpu核（num_workers）设置为4；batch_size设置为16
    - ![image-20240507192818047](C:\Users\www14\AppData\Roaming\Typora\typora-user-images\image-20240507192818047.png)
  - batch_size在上述基础上改为128,看能不能更加充分利用gpu
    - ![image-20240507193509601](C:\Users\www14\AppData\Roaming\Typora\typora-user-images\image-20240507193509601.png)
  - 再改为1024
    - ![image-20240507194038915](C:\Users\www14\AppData\Roaming\Typora\typora-user-images\image-20240507194038915.png)，没快多少…
  - **test_accuracy:**![image-20240507194228037](C:\Users\www14\AppData\Roaming\Typora\typora-user-images\image-20240507194228037.png)
- CNN_basic（1层）
  - test_accuracy:![image-20240507194352139](C:\Users\www14\AppData\Roaming\Typora\typora-user-images\image-20240507194352139.png)
- FCN_dropout
  - ![image-20240507202115893](C:\Users\www14\AppData\Roaming\Typora\typora-user-images\image-20240507202115893.png)

- FCN_hiddenlayers&num_of_nn
  - ![image-20240507211535596](C:\Users\www14\AppData\Roaming\Typora\typora-user-images\image-20240507211535596.png)
- LSTM：
  - 

![image-20240508213120379](C:\Users\www14\AppData\Roaming\Typora\typora-user-images\image-20240508213120379.png)

- 如果有时间，可以试一下gpt4说的rnn每次送一个像素（rgb3维）的方式，准确率多少



- Vanila_RNN
  - ![image-20240508215505559](C:\Users\www14\AppData\Roaming\Typora\typora-user-images\image-20240508215505559.png)