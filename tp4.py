#methode de Newton
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Calcul de f(x)
def f(x1,x2):
    return (x1 - 2)**4 + (x1 - 2*x2)**2

# Calcul des dérivées partielles de f(x)
def df(x1,x2):
    df_dx1 = 4 * (x1 - 2)**3 + 2 * (x1 - 2*x2)
    df_dx2 = -4 * (x1 - 2*x2)
    return (df_dx1, df_dx2)
# Calcul d(df(x))
def ddf(x1):
    ddd=np.array([
        [12*((x1-2)**2)+2,-4],
        [-4,8]
    ])
    return ddd


# Calcul la norme de df(x)
def Ndf(x,y):
    a,b=df(x,y)
    return np.sqrt(a**2+b**2)

def Norm(x,y):
    return np.sqrt(x**2+y**2)


# methode de Newton
def newton(x1,y2,eps=1e-2):
    x=[]
    x.append((x1,y2))
    k=0
    dddd=[[1000],[1000]]
    while(abs(Norm(dddd[0][0],dddd[1][0]))>eps):
        xx,xy=x[k]
        ddfi=np.linalg.inv(ddf(xx))
        dff=np.array(df(xx,xy)).reshape(2,1)
        xxx=np.array(x[k]).reshape(2,1)
        dddd=np.dot(ddfi,dff)
        xkp1=xxx-np.dot(ddfi,dff)
        x.append((xkp1[0][0],xkp1[1][0]))
        k+=1
        if(k>=1000):
            print("error")
            break
    print(x[-1])
    print(k)  
    return x  


#dessin    
x=newton(0,4)    
xx=[]
xy=[]
for i in range(len(x)):
    xx.append(x[i][0])
    xy.append(x[i][1])

Nx1, Nx2 = 100,100
xx1,xx2 = np.linspace(-0.1,2.5,Nx1),np.linspace(-0.1,4.1,Nx2)
X,Y=np.meshgrid(xx1,xx2)
Z=f(X,Y)
fig,ax= plt.subplots()
plt.contour(X,Y,Z,levels=25)
ax.plot(xx,xy,'ro')
ax.plot(xx,xy)
ax.plot(2,1,'o',color='orange',linewidth=5,label='la solution analytique')
ax.plot([2,-0.1],[1,1],color='orange', linestyle='--')
ax.plot([2,2],[1,-0.1],color='orange', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Courbes de niceau de f(x,y)')
ax.legend()
plt.show()

       