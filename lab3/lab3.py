import random as rnd
import matplotlib.pyplot as plt

x_bounds = [(0,1), (1,2), (2,3), (3,4)]
y_bounds = [(0,3), (1,3), (1,2), (0,2)]

def generateRandomPoint(x_min, y_min, x_max,y_max):
    x = rnd.uniform(x_min,x_max)
    y = rnd.uniform(y_min,y_max)
    return (x,y)

def check(x,y):
    for i in range(0,len(x_bounds)):
        if((x >= x_bounds[i][0] and x < x_bounds[i][1]) and (y >= y_bounds[i][0] and y < y_bounds[i][1])):
            return True
    return False
                
numOfIterations = 1000000
xmin, xmax = 0, 4
ymin, ymax = 0, 4
x_in, y_in = [], []
x_out,y_out = [], []
counter = 0

for i in range(0,numOfIterations):
    x,y = generateRandomPoint(xmin,ymin,xmax,ymax)
    inside = check(x,y)
    if(inside):
        counter += 1
        x_in.append(x)
        y_in.append(y)
    else:
        x_out.append(x)
        y_out.append(y)

area = (counter/numOfIterations)*(xmax - xmin) * (ymax - ymin)
print("given boundaries:")
for i in range(0,len(x_bounds)):
    print(f"{y_bounds[i][0]} <= y < {y_bounds[i][1]} when {x_bounds[i][0]} <= x < {x_bounds[i][1]}")

print(f"Area of given figure = {area}")

plt.scatter(x_in,y_in,color = 'cyan',s = 3)
plt.scatter(x_out,y_out,color = 'purple', s = 3)
plt.show()
