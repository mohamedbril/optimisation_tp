#methode de gradient a pas fix
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

# Calcul la direction
def direction(x,y):
    dx , dy = df(x,y)
    return (-dx , -dy)

# Calcul la norme de df(x)
def Ndf(x,y):
    a,b=df(x,y)
    return np.sqrt(a**2+b**2)

#test d'arret 2

def Norm(x,y):
    return np.sqrt(x**2+y**2)

# methode de gradient a pas fix
def gardient_pas_fix(x1,y2,alph=1e-1,eps=1e-2):
    x=[]
    x.append((x1,y2))
    k=0
    dx,dy=10000,10000
    #abs(Ndf(x[k][0],x[k][1]))>eps
    while(abs(Norm(dx*alph,dy*alph))>eps):
        dx,dy=direction(x[k][0],x[k][1])
        xx,xy=x[k]
        xxx=xx+alph*dx
        xxy=xy+alph*dy
        x.append((xxx,xxy))
        k+=1
        if(k>=1000):
            print("error")
            return None
    print(x[-1])
    print(k)  
    return x  


#dessin    
x=gardient_pas_fix(1,0)    
xx=[]
xy=[]
for i in range(len(x)):
    xx.append(x[i][0])
    xy.append(x[i][1])

Nx1, Nx2 = 100,100
xx1,xx2 = np.linspace(0.8,2.1,Nx1),np.linspace(-0.1,1.1,Nx2)
X,Y=np.meshgrid(xx1,xx2)
Z=f(X,Y)
fig,ax= plt.subplots()
plt.contour(X,Y,Z,levels=50)
ax.plot(xx,xy,'ro')
ax.plot(xx,xy)
ax.plot(2,1,'o',color='orange',linewidth=5,label='la solution analytique')
ax.plot([2,0.8],[1,1],color='orange', linestyle='--')
ax.plot([2,2],[1,-0.1],color='orange', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Courbes de niceau de f(x,y)')
ax.legend()
plt.show()

       
