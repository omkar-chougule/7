import time
from matplotlib import pyplot as plt

ar=[]
def iter_fibo(n):
    if(n<=0):
        return 0
    elif(n==1):
        return 1
    
    else:
        prev=0
        next=1
        print ("[0, 1,",end=" ")
        for i in range(2,n+1):
            prev , next = next , next + prev
            if(next>n):
                break
            print (next,end=", ")
        print("]")
        return next
def recursive_fibo(n):
    if (n<=1):
        ar[n]=n
        return n
    
    next = recursive_fibo(n-1)+recursive_fibo(n-2)
    ar[n] = next
    return next
n = 10
ar = [0 for i in range (n)]
# print_fibonacci_series(n)
t=time.time()
s=iter_fibo(n)
t2=time.time()
print(t2-t)
t=time.time()
s=recursive_fibo(n-1)
t2=time.time()
print (t2-t)
print(ar)