import sys
sys.path.append('../')
import numpy as np
from helper_func import incIndVec
from collections import deque # efficient implementation of queue (O(1) from both ends)


# search space znamo generirati, treba je narest bfs nad tem + rezat veje, ce je manj kot v prejsnjem levelu
def searchSpace(v: np.array, n: int):
    if(n >= len(v)):
        # print(v)
        return
    
    for i in range(len(v)-n):
        v_i = n+i
        v[v_i] = 1
        print(v)
        searchSpace(v, v_i+1)
        v[v_i] = 0


# bfs variant of ^
def searchSpace2(v: np.array, n: int):
    # 1. init queue
    queue = deque([(v,0)])
    n_el = len(v) # length of vector - number of elements

    # 2. loop
    while(len(queue) != 0):
        
        V, n = queue.popleft()
        
        for i in range(n_el-n):
            v_i = n+i
            V[v_i] = 1
            print(V)
            # searchSpace(v, v_i+1)
            queue.append((V.copy(), v_i+1))
            V[v_i] = 0


# test qbp matrix:
Q = np.array(
[[ 27.98888889,  -0.        ,  -13.99444444,  -0.         ,  -13.99444444,   -0.         ,  -13.99444444,  -0.        ,  -0.         ,  -13.99444444],
 [ -0.        ,  16.97777778,  -0.         ,  -0.         ,  -0.         ,   -0.         ,  -0.         ,  -0.        ,  -0.         ,  -0.        ],
 [-13.99444444,  -0.        ,  27.98888889 ,  -0.         ,  -13.99444444,   -0.         ,  -13.99444444,  -0.        ,  -0.         ,  -13.99444444],
 [ -0.        ,  -0.        ,  -0.         ,  25.92222222 ,  -0.         ,   -12.96111111,  -0.         ,  -0.        ,  -12.96111111,  -0.        ],
 [-13.99444444,  -0.        ,  -13.99444444,  -0.         ,  27.98888889 ,   -0.         ,  -13.99444444,  -0.        ,  -0.         ,  -13.99444444],
 [ -0.        ,  -0.        ,  -0.         ,  -12.96111111,  -0.         ,   25.92222222 ,  -0.         ,  -0.        ,  -12.96111111,  -0.        ],
 [-13.99444444,  -0.        ,  -13.99444444,  -0.         ,  -13.99444444,   -0.         ,  27.98888889 ,  -0.        ,  -0.         ,  -13.99444444],
 [ -0.        ,  -0.        ,  -0.         ,  -0.         ,  -0.         ,   -0.         ,  -0.         ,  19.44444444,  -0.         ,  -0.        ],
 [ -0.        ,  -0.        ,  -0.         ,  -12.96111111,  -0.         ,   -12.96111111,  -0.         ,  -0.        ,  25.92222222 ,  -0.        ],
 [-13.99444444,  -0.        ,  -13.99444444,  -0.         ,  -13.99444444,   -0.         ,  -13.99444444,  -0.        ,  -0.         ,  27.98888889]
])

# -----------------



# multibranch-ascent qbp solver:
# [TODO] - some testing needed (to "prove correctness", possible debugging needed), looks promising, however
def findMax(v: np.array, Q: np.array, n: int):
    # 1. init queue
    queue = deque([(v,0)])
    n_el = len(v) # length of vector - number of elements

    # max:
    D_max = 0
    v_max = v

    # 2. loop
    n_iter = 0
    while((len(queue) != 0)):
        
        V, n = queue.popleft()

        # check if current is better than any other from before, if it is, update&generate, else skip
        d_current = np.dot(V, np.dot(Q, V)) # to mogoce ne bo delal
        print(d_current)
        if(d_current < D_max):
            continue
        # if(d_current >= D_max):
        D_max = d_current
        v_max = V.copy()
        
        for i in range(n_el-n):
            v_i = n+i
            V[v_i] = 1
            # print(V)
            # searchSpace(v, v_i+1)
            queue.append((V.copy(), v_i+1))
            V[v_i] = 0
        n_iter += 1
    print("n_iter: " + str(n_iter))
    return (v_max, D_max)




def main():
    # v = np.array([0,0,0,0]).astype(np.uint8)
    v = np.zeros(10)
    # searchSpace2(v, 0)
    v_max, d_max = findMax(v,Q, 0)
    
    print("v_max: ",v_max, "d_max: ", d_max)
    
    # for i in range(10):
    #     incIndVec(v)
    #     print(v)
    

if __name__=="__main__":
    main()

