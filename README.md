# PAC-SerialVersion
This is the serial version code of our neural network. The code belongs to **广东工业大学爱国敬业队**.        
Team members: CHEN Qiguang (core developer), MAO Yadong, ZHENG Shiqiang, ZHAO Ganning.

**In *main.c*, we read data from files, feed the data into the neural network and train the model.**

**In *cnn.c*, we define the structure of our neural network.**
* The function *cnn_init* defines the structure of our model and its parameters.
* The function *go* performs a forward and backward propagation and gradient update.

**In *cnn.h*, we define some data structures.**               
- In the structure *CNN*:             
	- Pointer *\*layer_box stores* the pointer of each layer.
- In the structure *layer*:            
	- Pointer *\*x* is the input data.
	- Pointer *\*out* is the output data.
	- Pointer *\*dx* is the back propagation's output of the current layer.
	- Pointer *\*dout* is the back propagation's input of the current layer.
	- Other pointers are function pointers and they will be assigned when the model is initialized. The definitions of these function pointers vary according to the type of the layers.

**Each layer has a source file and a header file.**
* The source file defines the functions of initialization, forward and backward propagation and gradient update.
* The header file defines the data structure of the specific layer based on the layer structure defined in CNN.h.

**The file *read_npy.c* realizes reading data from *npy* format files.**
