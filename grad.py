x = 2
learning_rate = 0.1
precision = 0.000001
previous_step_size = 10
max_iterations = 50
iters = 0
gradient_function = lambda x: 2 * (x + 3)  # Derivative of (x + 3)^2

import matplotlib.pyplot as plt
gd = []

while precision < previous_step_size and iters < max_iterations:
    prev = x
    x = x - learning_rate * gradient_function(prev)
    previous_step_size = abs(x - prev)
    iters += 1
    print('iteration', iters, 'values', x)
    gd.append(x)

print('local minimum:', x)

# Optional: Plot the gradient descent progress
plt.plot(gd)
plt.xlabel('Iteration')
plt.ylabel('x values')
plt.title('Gradient Descent Progress')
plt.show()
import numpy as np
def f(x):
    return (x+3)**2
x = np.linspace(-10,10,100)
y = f(x)

plt.plot(x,y)

history_y = [f(i) for i in gd]

plt.scatter(gd,history_y)
plt.show()