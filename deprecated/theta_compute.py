#计算倾斜角

import math

fy = 6.46306821e+03
y1 = 1791 - 1080
y2 = 1879-1080
X = 0.174
Y = 1.3
x1 = 3.77

b = math.atan(y1/fy)
c = math.atan(y2/fy)
temp = math.cos(b-c)-2*Y*math.sin(b-c)/X
theta = math.pi-(math.acos(temp)+b+c)/2
theta = 6.7*math.pi/180
res = (math.tan(math.pi/2-theta-c)-math.tan(math.pi/2-theta-b)) * Y
print(f"result theta: {theta*180/math.pi}")
print(f"{X},{res}")
