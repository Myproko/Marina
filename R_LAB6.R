MLE =numeric (1000)
CI_L=numeric(m)
CI_U=numeric(m)
counter =  
for (i in 1:1000)
{[=runif (20,0,5)
  y = sort (x)
  MLE[i]=y[20]
  CI_L[i] = MLE [i]/qbeta(0.95, 20,1)
  CI_U[i] = MLE[i]/qbeta(0.05, 20,1)
  if ((CI_L[i] ,5) & (CI_U[i] ))}
  
  
  
  