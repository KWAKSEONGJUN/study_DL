from add_layer import AddLayer
from mul_layer import MulLayer

apple = 100
apple_num = 2
tax = 1.1
orange = 150
orange_num = 3

apple_mul_layer = MulLayer()
orange_mul_layer = MulLayer()
add_layer = AddLayer()
tax_mul_layer = MulLayer()

#순전파
apple_price = apple_mul_layer.forward(apple, apple_num)
orange_price = orange_mul_layer.forward(orange, orange_num)

price = add_layer.forward(apple_price, orange_price)
price = tax_mul_layer.forward(price, tax)
print(price)

#역전파
dprice = 1
dprice, dtax = tax_mul_layer.backward(dprice)
dapple_price, dorange_price = add_layer.backward(dprice)
dapple_price, dapple_num = apple_mul_layer.backward(dapple_price)
dorange_price, dorange_num = orange_mul_layer.backward(dorange_price)

print(dapple_price, dapple_num, dorange_price, dorange_num, dtax)


