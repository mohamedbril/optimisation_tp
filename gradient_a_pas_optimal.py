#methode de gradient a pas optimal
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


def g(x,y,ro):
    dirc=direction(x,y)
    return f(x+np.multiply(dirc[0],ro),y+np.multiply(dirc[1],ro))

def dg(x,y,ro):
    dirc=direction(x,y)
    return df(x+np.multiply(dirc[0],ro),y+np.multiply(dirc[1],ro))

#search liner méthode de Wolfe
def wolfe(x,y,roi):
    dirc=direction(x,y)
    m1=0.1
    m2=0.7
    rop=np.Inf
    rom= 0
    ro=[]
    ro.append(roi)
    k=0
    while((g(x,y,ro[k])>=(f(x,y)+np.multiply(m1,np.dot(np.array(df(x, y)).reshape(2, 1).T,dirc))))and(np.dot(np.array(dg(x,y,ro[k])).reshape(2,1).T,dirc)<=m2*np.dot(np.array(df(x, y)).reshape(2, 1).T,dirc))):
        if(k>=100):
            break
        if((g(x,y,ro[k]))>=(f(x,y)+np.multiply(m1,np.dot(np.array(df(x, y)).reshape(2, 1).T,dirc)))):
            rop=ro[k]
            ro.append((rop+rom)/2)
        else :
            rom=ro[k]
            if(rop<np.Inf):
                ro.append((rop+rom)/2)
            else:
                ro.append(2*ro[k])        
        k+=1

    return ro[-1]

def Norm(x,y):
    return np.sqrt(x**2+y**2)

# methode de gradient a pas optimal
def gardient_pas_optimal(x1,y2,eps=1e-2):
    x=[]
    x.append((x1,y2))
    k=0
    pas=0.1
    dx,dy=10000,10000
    while(abs(Norm(dx*pas,dy*pas))>eps):
        dx,dy=direction(x[k][0],x[k][1])
        xx,xy=x[k]
        pas=wolfe(x[k][0],x[k][1],0.1)
        xxx=xx+pas*dx
        xxy=xy+pas*dy
        x.append((xxx,xxy))
        k+=1
        if(k>=1000):
            print("error")
            break
    print(x[-1])
    print(k)  
    return x  


#dessin    
x=gardient_pas_optimal(4,0)    

xx=[]
xy=[]
for i in range(len(x)):
    xx.append(x[i][0])
    xy.append(x[i][1])

Nx1, Nx2 = 100,100
xx1,xx2 = np.linspace(-0.5,4.5,Nx1),np.linspace(-0.1,2,Nx2)
X,Y=np.meshgrid(xx1,xx2)
Z=f(X,Y)
fig,ax= plt.subplots()
plt.contour(X,Y,Z,levels=25)
ax.plot(xx,xy,'ro')
ax.plot(xx,xy)
ax.plot(2,1,'o',color='orange',linewidth=5,label='la solution analytique')
ax.plot([2,-0.5],[1,1],color='orange', linestyle='--')
ax.plot([2,2],[1,-0.1],color='orange', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Courbes de niceau de f(x,y)')
ax.legend()
plt.show()

       
