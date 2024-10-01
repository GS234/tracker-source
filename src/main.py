from helper_func import coords2det2
from DetectionSpace import DetectionSpace
from Trajectory import Trajectory

# testing trajectories:
t1 = [
    [(122, 52)],
    [(115, 64)],
    [(120, 79),(112, 81),(116, 83),(116, 84),(114, 87)],
    [(110, 98),(104, 103),(105, 105),(105, 106),(101, 109),],
    [(104, 126),(99, 128) ,(98, 129) ,(99, 129) ,(100, 132)],
    [(114, 146),(114, 147)],
    [(128, 173),(131, 174),(132, 174)],
    [(148, 191),(152, 191)],
    [(146, 207),(145, 210),(148, 210),(145, 211),(148, 211)],
    [(146, 226)],
    [(141, 232)],
    [(133, 232),(133, 234),(133, 236)],
    [(123, 234)],
    [(114, 224)],
    [(103, 210),(103, 211),(104, 212)],
    [(96, 215)],
    [(89, 223),(89, 226),(87, 228),(91, 228),(91, 230),(90, 231)],
    [(93, 242)],
    [(105, 256)],
    [(113, 270),(113, 272)],
    [(118, 285)],
    [(123, 287)],
    [(134, 284)],
    [(145, 294),(143, 295),(146, 296)],
    [(158, 312)],
    [(156, 326)],
    [(160, 338),(160, 339)],
    [(155, 352),(155, 354)],
    [(150, 364)],
    [(144, 373),(143, 374),(142, 375),(140, 376)],
    [(126, 384)],
    [(109, 387),(108, 388),(110, 389)],
    [(97, 398)],
    [(96, 408),(96, 409)],
    [(85, 424),(85, 425),(84, 426),(85, 426)],
    [(80, 439)],
    [(94, 446)],
    [(116, 452),(117, 452)],
    [(139, 440),(137, 441),(136, 442),(137, 442)],
    [(149, 439)],
    [(163, 436)],
    [(188, 426)],
    [(204, 409)]
]

t3 = [
    [(275, 114)],
    [(265, 135), (266, 135), (266, 136), (266, 137)],
    [(273, 152), (274, 152)],
    [(282, 170)],
    [(283, 194)],
    [(291, 214), (292, 214)],
    [(306, 224)],
    [(313, 236), (314, 237), (314, 238)],
    [(314, 254), (314, 255), (314, 256), (314, 257)],
    [(304, 277), (304, 278), (303, 279), (301, 280), (302, 280), (300, 281), (304, 281), (305, 289)],
    [(287, 296)],
    [(277, 309), (277, 310)],
    [(286, 319), (287, 320), (288, 320), (288, 321), (289, 321)],
    [(306, 330), (306, 331), (306, 332)],
    [(297, 349), (298, 349)],
    [(303, 362)],
    [(289, 374), (290, 374), (287, 375), (288, 375), (286, 376)],
    [],
    [],
    [],
    [(255,420)]
]



t2 = [
    [(250,249)],
    [(240,240),(260,240)],
    [(240,240),(260,240),(253,240),(256,240)]
]

t4 = [
[(262, 201)],
[(254, 206)],
[(247, 215)],
[(238, 221),(242, 226),(230, 210)],
[(229, 233)],
[(219, 245)],
[(207, 253)],
[(202, 262)],
[(198, 269)],
[(193, 274)],
[(182, 280)]
]

tdeb_1 = [
    [(250,250)], # 0
    [(255,255)], # 1
    [(260,255),(265,260)], # 2
    [(265,265)], # 3
    [],
    [(270,270)], # 4
]

t3_b = [
[(168, 224), (262, 201)],
[(179, 233), (254, 206)],
[(188, 241), (247, 215)],
[(195, 242), (238, 221),(242, 226)],
[(203, 242), (229, 233)],

[(209, 247), (219, 245)],
[(217, 254), (207, 253)],

[(225, 258), (202, 262)],
[(232, 262), (198, 269)],
[(238, 268), (193, 274)],
[(249, 275), (182, 280)],
[(263, 272)]
]

# ---------------------

# main:
# def main():
#     n = 500 # canvas size
#     seed = 42
    
#     D = coords2det2(t3_b)

#     # create detection space object:
#     dspace = DetectionSpace(n)
#     dspace.D = D

#     d0 = D[0][1]
#     # d1 = D[1][0]

#     th1 = Trajectory(d0, dspace)
#     th1.build()
#     th1.drawToSpace()
    
#     # th2 = Trajectory(d1, dspace)
#     # th2.build()
#     # th2.drawToSpace()
    
#     # th3 = Trajectory(d2, dspace)
#     # th3.build()
#     # th3.drawToSpace()
    
#     # th4 = Trajectory(d3, dspace)
#     # th4.build()
#     # # th4.drawToSpace()
#     # print(th1.X)


#     Q = dspace.buildQBPMatrix([th1, th1], 0.0, 1.0)
#     print("Q:\n", Q)
#     v = dspace.solveQBP(Q)
#     print(v)
        
#     dspace.showSpace()


def main():
    n = 500 # canvas size
    seed = 42
    
    D = coords2det2(t3_b)

    det_list = []
    for dl in D:
        for d in dl:
            det_list.append(d)
    print(det_list)

    # create detection space object:
    dspace = DetectionSpace(n)
    dspace.D = D

    tr = []

    for i in range(0,len(det_list)-15, 1):
        d = det_list[i]
        new_tr = Trajectory(d, dspace)
        new_tr.build()
        # new_tr.drawToSpace()
        tr.append(new_tr)



    # d0 = D[0][0]
    # d1 = D[0][1]
    # d2 = D[6][0]
    

    # th1 = Trajectory(d0, dspace)
    # th1.build()
    # th1.drawToSpace()
    
    # th2 = Trajectory(d1, dspace)
    # th2.build()
    # th2.drawToSpace()
    
    # th3 = Trajectory(d2, dspace)
    # th3.build()
    # th3.drawToSpace()
    
    # th4 = Trajectory(d3, dspace)
    # th4.build()
    # # th4.drawToSpace()


    # Q = dspace.buildQBPMatrix([th1, th2, th3, th4], 1.0, 1.0)
    # Q = dspace.buildQBPMatrix([th1, th2], 1.0, 1.0)
    # Q = dspace.buildQBPMatrix([th1, th2, th3], 1.0, 1.0)
    Q = dspace.buildQBPMatrix(tr, 1.0, 0.1)
    # Q = dspace.buildQBPMatrix([tr[0],tr[1],tr[4],tr[5]], 0.0, 0.1)
    # Q = dspace.buildQBPMatrix([tr[0], tr[4]], 0.0, 0.1)


    # Q = dspace.buildQBPMatrix([tr[1],tr[3]], 1.0, 1.0)
    
    print("Q:\n", Q)
    v = dspace.solveQBP(Q)
    print(v)
    
    vv = v
    for i in range(len(tr)):
        if(vv % 2 != 0):
            tr[i].drawToSpace()
        vv = vv // 2
    
    # # tr[2].drawToSpace()
    
    # tr[0].drawToSpace()
    # tr[1].drawToSpace()
    
    # tr[4].drawToSpace()
    # tr[5].drawToSpace()
    
    dspace.showSpace()


if __name__ == "__main__":
    main()