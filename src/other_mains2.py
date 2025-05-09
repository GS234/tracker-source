import numpy as np
from helper_func import *

def mainInitSelect(n: int, options: dict):
    global OPTIONS
    OPTIONS = options
    if(n == 6):
        main6()

def main6():
    print("this is main6 from other_mains2")
    print("options: ")
    print(OPTIONS)

    # _, _, all_trs = readTrajectoryFile("./tracker/all_trs.p")
    # all_trs = list(all_trs.values())
    # printTrListWithStatsOrdered(all_trs)
    # print()

    # trs = readTrajectoryFile('../tr.p')
    # trs_selected = buildGroupSolve([])
    # trs_selected = buildGroupSolve(trs)
    # print(trs_selected)
    # aaa = []
    # for aa in a:
    #     aaa = aaa+aa
    # aaa.sort()
    # print(aaa)

    # pure, not_pure = analyzeTrsWithQ(trs)
    # # print(pure)
    # print("pure: ")
    # printTrListWithStatsOrdered(pure)
    # print("not pure: ")
    # printTrListWithStatsOrdered(not_pure)
    # Q = buildQBPMatrixX(not_pure, type=1)
    # np.savetxt("q.txt",Q, fmt="%7.2f")
    # print(Q)

    # Q = np.array([
    #     #1 2 3 4 5 6 7 8 9 10
    #     [1,1,0,0,0,0,0,0,0,0], # 1
    #     [1,1,1,0,0,0,0,0,0,0], # 2
    #     [0,1,1,0,0,0,0,0,0,0], # 3
    #     [0,0,0,1,0,0,0,0,1,0], # 4
    #     [0,0,0,0,1,1,0,0,0,1], # 5
    #     [0,0,0,0,1,1,0,0,0,1], # 6
    #     [0,0,0,0,0,0,1,0,1,0], # 7
    #     [0,0,0,0,0,0,0,1,0,1], # 8
    #     [0,0,0,1,0,0,1,0,1,0], # 9
    #     [0,0,0,0,1,1,0,1,0,1], # 10
    #     ])
    # Q = np.array(
    #     [[ 4.557, -0.   , -4.347, -0.   , -3.446, -0.   , -2.385, -2.01 , -0.   , -1.554, -0. ,   -0.6  ],
    #      [-0.   ,  3.711, -0.   , -3.526, -0.   , -2.543, -0.   , -0.   , -1.593, -0.   , -0.6,   -0.   ],
    #      [-4.347, -0.   ,  3.623, -0.   , -3.446, -0.   , -2.385, -2.01 , -0.   , -1.554, -0. ,   -0.6  ],
    #      [-0.   , -3.526, -0.   ,  2.938, -0.   , -2.543, -0.   , -0.   , -1.593, -0.   , -0.6,   -0.   ],
    #      [-3.446, -0.   , -3.446, -0.   ,  2.872, -0.   , -2.385, -2.01 , -0.   , -1.554, -0. ,   -0.6  ],
    #      [-0.   , -2.543, -0.   , -2.543, -0.   ,  2.119, -0.   , -0.   , -1.593, -0.   , -0.6,   -0.   ],
    #      [-2.385, -0.   , -2.385, -0.   , -2.385, -0.   ,  1.987, -2.01 , -0.   , -1.554, -0. ,   -0.6  ],
    #      [-2.01 , -0.   , -2.01 , -0.   , -2.01 , -0.   , -2.01 ,  1.675, -0.   , -1.554, -0. ,   -0.6  ],
    #      [-0.   , -1.593, -0.   , -1.593, -0.   , -1.593, -0.   , -0.   ,  1.328, -0.   , -0.6,   -0.   ],
    #      [-1.554, -0.   , -1.554, -0.   , -1.554, -0.   , -1.554, -1.554, -0.   ,  1.295, -0. ,   -0.6  ],
    #      [-0.   , -0.6  , -0.   , -0.6  , -0.   , -0.6  , -0.   , -0.   , -0.6  , -0.   ,  0.5,   -0.   ],
    #      [-0.6  , -0.   , -0.6  , -0.   , -0.6  , -0.   , -0.6  , -0.6  , -0.   , -0.6  , -0. ,    0.5  ]])
    Q = np.array([
        [ 3.419, -3.134, -2.348, -1.698, -0.959, -0.6,  ],
        [-3.134,  2.612, -2.348, -1.698, -0.959, -0.6,  ],
        [-2.348, -2.348,  1.957, -1.698, -0.959, -0.6,  ],
        [-1.698, -1.698, -1.698,  1.415, -0.959, -0.6,  ],
        [-0.959, -0.959, -0.959, -0.959,  0.799, -0.6,  ],
        [-0.6  , -0.6,   -0.6,   -0.6,   -0.6,    0.5,  ]])
    print(Q)
    # a = getGroupsFromQ(Q)
    a = buildGroupSolve([1],1)
    print("without groups:")
    res = solveQBP2(Q)
    print(res)
    print("solving with groups:")
    print(a)


# solving (merge) ...
# [[ 3.419 -3.134 -2.348 -1.698 -0.959 -0.6  ]
#  [-3.134  2.612 -2.348 -1.698 -0.959 -0.6  ]
#  [-2.348 -2.348  1.957 -1.698 -0.959 -0.6  ]
#  [-1.698 -1.698 -1.698  1.415 -0.959 -0.6  ]
#  [-0.959 -0.959 -0.959 -0.959  0.799 -0.6  ]
#  [-0.6   -0.6   -0.6   -0.6   -0.6    0.5  ]]
# solving QBP (6x6)
# [buildGroupSolve] solving with groups
# group 1 [0, 1, 2, 3, 4]:
# [[ 3.419 -3.134 -2.348 -1.698 -0.959]
#  [-3.134  2.612 -2.348 -1.698 -0.959]
#  [-2.348 -2.348  1.957 -1.698 -0.959]
#  [-1.698 -1.698 -1.698  1.415 -0.959]
#  [-0.959 -0.959 -0.959 -0.959  0.799]]
# solving Q ...
# solving QBP (5x5)
# (array([1, 0, 0, 0, 0], dtype=uint8), 3.418967849300367)
# [0]
# group 2 [5]:
# [[0.5]]
# solving Q ...
# solving QBP (1x1)
# (array([1], dtype=uint8), 0.5)
# [5]
# done solving, v:
# (array([1, 0, 0, 0, 0, 0], dtype=uint8), 3.418967849300367)
# v2:
# (array([1, 0, 0, 0, 0, 1], dtype=uint8), 3.918967849300367)
# adding selected merged to idTr_map
# added
# [@ 122 / 4101 (2%)]
# [starting new]
# [UPDATING TARGET]
# target lost, searching for new based on visual similarity:
