class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out
    
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


# apple = 100
# apple_num = 2
# tax = 1.1

# mul_apple_layer = MulLayer()
# mul_tax_layer = MulLayer()

# #순전파
# price = mul_tax_layer.forward(mul_apple_layer.forward(apple, apple_num), tax)
# print(price)

# #역전파
# dprice = 1
# dprice, dtax = mul_tax_layer.backward(dprice)
# dprice, dapple_num = mul_apple_layer.backward(dprice)
# print(dprice, dapple_num, dtax)

