import numpy as np


# 公式:
#   train:  output = input * (mask / (1 - p))
#   test:   output = input
class Dropout:
    def __init__(self, p):
        self.p = p

    def __call__(self, input, mode):
        return self.forward(input, mode)

    def forward(self, input, mode):
        if mode != "train":
            return input
        self.mask = np.random.choice([0, 1], size=input.shape, p=[1-self.p, self.p])

        return self.mask * input / (1 - self.p)
    
    def backward(self, d_out):

        # y = mask * x
        # dy/dx = mask
        return d_out * self.mask
    
if __name__ == "__main__":
    input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    dropout_layer = Dropout(p=0.5)
    
    train_output = dropout_layer(input_data, mode="train")
    print(train_output)
    
    inference_output = dropout_layer(input_data, mode="test")
    print(inference_output)
