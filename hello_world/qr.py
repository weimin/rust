import numpy as np;
import scipy.linalg as sp;
import math;

tol = 1e-16;
#A = np.random.rand(5,2);
A = np.array([[1,0,0],[1,1,1],[1,2,4],[1,3,9],[1,4,16]]);
b = np.array([-1,7,2]);
Q,R = sp.qr(A);
print(A);
print(Q);
print(R);
#newcol = np.arange(1,len(A)+1);
newcol = np.array([0,1,8,27,64]);
newcol = np.reshape(newcol,(-1,1));
A1 = np.hstack((A,newcol));
print(A1);
R1 = Q.T@A1;
#R1[abs(R1)<tol] = 0;
print('R1:',R1);
def givens(a,b):
  c = 0;
  s = 0;
  if(b==0):
    c = 1;
    s = 0;
  else:
    if(abs(b)>=abs(a)):
      t = -a/b;
      s = 1/math.sqrt(1+t**2);
      c = s*t;
    else:
      t = -b/a;
      c = 1/math.sqrt(1+t**2);
      s = c*t;
  return c,s;
u = R1[:,R1.shape[1]-1];
print('u:',u);
n = len(u);
c,s = givens(u[n-2],u[n-1]);
Givens = np.array([[c,s],[-s,c]]);
#u[n-2] = c*u[n-2]-s*u[n-1];
#u[n-1] = 0;
#u = np.reshape(u,(-1,1));
#R2 = np.hstack((R,u));
# Update R
i = n-2;
j = n-1;
#tmp = c*R1[i,:] - s*R1[j,:];
#R1[j,:] = s*R1[i,:]+c*R1[j,:];
#R1[i,:] = tmp;
R1[i:,i:] = Givens.T@R1[i:,i:];
print('R1hat:',R1);
# Update Q
Q2 = Q;
Q2[:,i:] = Q2[:,i:]@Givens; 
Q2ref,R2ref = sp.qr(A1);
print('R2ref',R2ref);
