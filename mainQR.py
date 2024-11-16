import numpy as np
import biblioteca as bib
import scipy.linalg as la
 
#A = np.array([[2.,-3.,5.],[6.,-1.,3.],[0.,1.,0.]],float) 
A = np.random.uniform(-10,10,(5,5))
QR = bib.QRdecomp(A)
Q = QR[0]
R = QR[1]

#print("QtQ: ",Q.transpose()@Q)
#R = Q.transpose()@A
#print("R: ",R)

print("A-QR: ",A-Q@R)