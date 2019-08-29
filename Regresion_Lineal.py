import numpy as np
import matplotlib.pyplot as plt 

from sklearn.datasets import load_boston
## Carga Data Set
boston = load_boston()

print (boston.DESCR)

X= np.array(boston.data[:,5])
Y= np.array(boston.target)

plt.scatter(X,Y, alpha=0.3)

## Añadimos columna de 1s para termino independiente 
X = np.array([np.ones(506),X]).T

B = np.linalg.inv(X.T @ X) @ X.T @ Y 

plt.plot([4,9],[B[0] + B[1] * 4, B[0] + B[1] * 9], c='red') 
plt.show
