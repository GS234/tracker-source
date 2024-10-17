# multibranch ascent algorithm sandbox
import numpy as np


# helper func:
def incIndVec(vec: list[int], rev=False):
    c = 1
    n = len(vec)

    rev_i = 1
    if(rev):
        rev_i = 0

    for i in range(n):
        # i_i = rev_i*(n-1-i) + (1-rev_i)*i # branchless
        i_i = rev_i*(n-1-(i<<1))+i # branchless, optimized :)

        vec[i_i] = vec[i_i] + c
        if(vec[i_i] == 2):
            vec[i_i] = 0
            c = 1
        else:
            break
# ------------

Q = np.array(
    [[11.85111111, -0.91944444, -5.92555556, -0.91944444, -5.92555556, -0.91944444, -5.92555556, -0.92166667, -0.96888889],
     [-0.91944444, 10.81333333, -0.91944444, -5.40666667, -0.91944444, -5.40666667, -0.91944444, -5.42,       -5.46777778],
     [-5.92555556, -0.91944444, 11.85333333, -0.91944444, -5.92666667, -0.91944444, -5.92666667, -0.92166667, -0.96888889],
     [-0.91944444, -5.40666667, -0.91944444, 10.81333333, -0.91944444, -5.40666667, -0.91944444, -5.42,       -5.46777778],
     [-5.92555556, -0.91944444, -5.92666667, -0.91944444, 11.85333333, -0.91944444, -5.92666667, -0.92166667, -0.96888889],
     [-0.91944444, -5.40666667, -0.91944444, -5.40666667, -0.91944444, 10.81333333, -0.91944444, -5.42,       -5.46777778],
     [-5.92555556, -0.91944444, -5.92666667, -0.91944444, -5.92666667, -0.91944444, 11.85444444, -0.92166667, -0.96888889],
     [-0.92166667, -5.42,       -0.92166667, -5.42,       -0.92166667, -5.42,       -0.92166667,  9.84,       -4.95611111],
     [-0.96888889, -5.46777778, -0.96888889, -5.46777778, -0.96888889, -5.46777778, -0.96888889, -4.95611111,  9.93555556]
    ])

Q_red = np.array(
    [[11.85111111, -0.91944444, -5.92555556, -0.91944444,],
     [-0.91944444, 10.81333333, -0.91944444, -5.40666667,],
     [-5.92555556, -0.91944444, 11.85333333, -0.91944444,],
     [-0.91944444, -5.40666667, -0.91944444, 10.81333333,],
    ]
)

# function creates array that has binary representation of integer
def binArrFromInt(n: int, l: int):
    i = 0
    v = np.zeros(l).astype(np.int8)
    while (n != 0 and i < l):
        v[i] = n%2
        n = n>>1
        i = i+1
    return v

# in-place version of ^
def setBinArrayToN(n:int, v: np.array):
    i = 0
    # v = v*0
    # print(len(v))
    # v[:] = 0
    while (n != 0 or i < len(v)):
        # print(n%2)
        v[i] = n%2
        n = n>>1
        i = i+1



# multibranch ascent
def mb(Q):
    # 1. start without trajectories
    v = np.zeros(np.shape(Q)[0]).astype(np.uint8)
    # print(v)

    # 2. R = 1 (vsako posebi)
    print("level 1: single trajectories")
    b_i = 1
    for i in range(len(v)):
        setBinArrayToN(b_i, v)
        D = np.dot(v,np.dot(Q, v))
        print(v, " d: ", D)
        b_i = b_i << 1
    
    # 3. R = 2
    print("level 2: pairs")
    a = [1, 2, 4, 8]





# main function
def main():
    # print(Q)

    v = np.zeros(np.shape(Q)[0]).astype(np.uint8)
    # print(v)
    # incIndVec(v)
    # print(v)

    # for i in range(2**len(v)-1):
    #     print(np.dot(v, np.dot(Q, v)))
    #     incIndVec(v)

    # print("len: ", 2**len(v)-1)
    mb(Q_red)
    # print(v)
    # for i in range(10):
    #     setBinArrayToN(i, v)
    #     print(v)





if __name__ == "__main__":
    main()



