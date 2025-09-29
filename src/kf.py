from matplotlib import pyplot as plt
import numpy as np
from helper_func import *


class Kalman:
    uid = 0 # global id
    def __init__(self, H, F, Q, R):
        self.id = Kalman.uid
        self.H: np.ndarray = H if(H is not None) else np.zeros((1,1)) # observation matrix
        self.F: np.ndarray = F if(F is not None) else np.zeros((1,1)) # state change matrix
        
        self.Q: np.ndarray = Q if(Q is not None) else np.zeros((1,1)) # process noise
        self.R: np.ndarray = R if(R is not None) else np.zeros((1,1)) # measurement noise
        Kalman.uid = Kalman.uid+1
    
    # predict
    def predict(self, x_pred: np.ndarray, G: np.ndarray, u: np.ndarray) -> np.ndarray:
        # print(self.F)
        # print(x_pred)
        # print(G)
        # print(u)

        
        x = np.dot(self.F, x_pred) + np.dot(G, u)
        return x
    def covPredict(self, p_pred: np.ndarray) -> np.ndarray:
        # print("p_pred:")
        # print(p_pred)
        # print("self f:")
        # print(self.F.T)
        pf_t = np.dot(p_pred, self.F.T)
        # print("pf_t:")
        # print(pf_t)
        p = np.dot(self.F, pf_t) + self.Q
        return p


    # update
    def stateUpdate(self, x_pred: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
        # sizes: x: (m, 1), z: (n, 1), H: (n, m), K: (m, n)
        inno = z - np.dot(self.H, x_pred) # innovation: how far off is prediction from measurement (size: (n, 1))
        x_nn = x_pred + np.dot(K, inno) # update (size: (m, 1))
        return x_nn
    
    def covUpdate(self, P_pred: np.ndarray, K: np.ndarray) -> np.ndarray:
        n,_ = np.shape(K)
        id: np.ndarray = np.diag(np.ones(n))

        id_kh: np.ndarray = id - np.dot(K, self.H)
        a: np.ndarray = np.dot(id_kh, np.dot(P_pred, id_kh.T))
        b: np.ndarray = np.dot(K, np.dot(self.R, K.T))
        P_c = a + b
        return P_c

    def getK(self, P_pred):
        hphtr = np.dot(self.H, np.dot(P_pred, self.H.T)) + self.R
        hphtr_inv = np.linalg.inv(hphtr)
        K = np.dot(P_pred, np.dot(self.H.T, hphtr_inv))
        return K
    
    def __str__(self):
        ret_str = "model %d:\nF:\n %s\nH:\n %s\nQ:\n %s\nR:\n %s"%(self.id, str(self.F), str(self.H), str(self.Q), str(self.R))
        return ret_str



def predict(x):
    return x

def predict2(x_pr, v, dt):
    return x_pr+dt*v



def update(pred, measurement, gain):
    innovation = measurement-pred
    return update2(pred, innovation, gain)

def update2(pred, innovation, gain):
    return pred + gain*innovation

def main():
    # example5()
    example6()

def example6():
    # np.cov(x,y)
    i = 0
    # x_true = 50
    z_i = [49.03,48.44,55.21,49.98,50.6,52.61,45.87,42.64,48.26,55.84]

    # true temp: 50c
    # q = 0.0001 # assume accurate model: process noise variance q
    q = np.array([[0.15]])
    r = np.array([[0.01]]) # measurement error: 0.1c
    # dt = 5s
    h = np.array([[1]])
    f = np.array([[1]])

    kf = Kalman(h,f,q,r)
    zero_const = np.array([[0]])


    noisy_temp=[50.005, 49.994, 49.993, 50.001, 50.006, 49.998, 50.021, 50.005, 50, 49.997] # process noise: +- 0.01 ('true values')
    z_i = [49.986, 49.963, 50.09, 50.001, 50.018, 50.05, 49.938, 49.858, 49.965,50.114] # measurement noise: +- 0.1

    # iteration 0:
    
    # init
    x_ = 60 # init temp (we guess)
    p_ = 100*100 # guess is imprecise, so we set init estimate to 100c

    # predict:
    # x_1 = x_
    # p_1 = p_ + q

    x_1 = kf.predict(x_, zero_const, zero_const)
    p_1 = kf.covPredict(p_)
    
    i_fin = len(z_i)
    while(i < i_fin):
        z_1 = z_i[i]

        # current state update:
        # K = p_1 / (p_1+r) # kalman gain
        # x_ = x_1 + K*(z_1-x_1) # current state est.
        # p_ = (1-K)*p_1
        
        K = kf.getK(p_1)
        x_ = kf.stateUpdate(x_1, z_1, K)
        p_ = kf.covUpdate(p_1, K)
        
        # predict:
        x_1 = kf.predict(x_, zero_const, zero_const)
        p_1 = kf.covPredict(p_)
        
        print("[%d] current state: x: %s, p: %s\npredict: x: %s, p: %s"%((i+1),str(x_), str(p_), str(x_1), str(p_1)))
        
        i = i+1

def example5():
    # np.cov(x,y)
    i = 0
    # x_true = 50
    z_i = [49.03,48.44,55.21,49.98,50.6,52.61,45.87,42.64,48.26,55.84]

    # true temp: 50c
    # q = 0.0001 # assume accurate model: process noise variance q
    q = 0.15
    r = 0.01 # measurement error: 0.1c
    # dt = 5s

    noisy_temp=[50.005, 49.994, 49.993, 50.001, 50.006, 49.998, 50.021, 50.005, 50, 49.997] # process noise: +- 0.01 ('true values')
    z_i = [49.986, 49.963, 50.09, 50.001, 50.018, 50.05, 49.938, 49.858, 49.965,50.114] # measurement noise: +- 0.1

    # iteration 0:
    
    # init
    x_ = 60 # init temp (we guess)
    p_ = 100*100 # guess is imprecise, so we set init estimate to 100c

    # predict:
    x_1 = x_
    p_1 = p_ + q
    
    i_fin = len(z_i)
    while(i < i_fin):
        z_1 = z_i[i]

        # current state update:
        K = p_1 / (p_1+r) # kalman gain
        x_ = x_1 + K*(z_1-x_1) # current state est.
        p_ = (1-K)*p_1
        
        # predict:
        x_1 = x_
        p_1 = p_ + q
        
        print("[%d] current state: x: %.3f, p: %.4f\npredict: x: %.3f, p: %.4f"%((i+1),x_, p_, x_1, p_1))
        
        i = i+1

def example4():
    i = 0
    # x_true = 50
    z_i = [49.03,48.44,55.21,49.98,50.6,52.61,45.87,42.64,48.26,55.84]
    
    # iteration 0:
    
    # init
    x_ = 60 # init height
    p_ = 15*15 # init covariance

    # predict:
    x_1 = x_
    p_1 = p_
    
    i_fin = len(z_i)
    r1 = 25
    while(i < i_fin):
        z_1 = z_i[i]
        K = p_1 / (p_1+r1)

        # current state update:
        x_ = x_1 + K*(z_1-x_1)
        p_ = (1-K)*p_1
        
        # predict:
        x_1 = x_
        p_1 = p_
        
        print("current state: x: %.2f, p: %.2f\npredict: x: %.2f, p: %.2f"%(x_, p_, x_1, p_1))
        
        i = i+1


def example3():
    # measurements = [30171,30353,30756,30799,31018,31278,31276,31379,31748,32175]
    measurements = [30221,30453,30906,30999,31368,31978,32526,33379,34698,36275]
    a,b = 0.2,0.1
    # a,b = 0.8,0.5
    dt = 5

    # initial conditions:
    x_p = 30000 # initial position
    x_v = 50 # estimated velocity

    # predict:
    x_p = predict2(x_p, x_v, 5)
    x_v = x_v # stays the same
    print(x_p)
    
    # first iteration:
    i=1
    x_p_l = []
    x_predictions = []
    for n in range(len(measurements)):
    # for n in range(2):
        x_p
        x_v
        x_predictions.append(x_p)
        z_i = measurements[i-1]

        x_p_temp = x_p
        x_v_temp = x_v
        x_p = update(x_p_temp, z_i, a)
        x_v = update2(x_v_temp, (z_i-x_p_temp)/dt, b)
        x_p_l.append(x_p)
        
        print("%.2f\n%.2f\n"%(x_p, x_v))
        x_p = x_p + dt*x_v
        x_v = x_v
        i = i+1
    
    dt_l = np.arange(len(measurements))*dt
    x_l = (np.arange(len(measurements))+1)*(dt*x_v)+30000
    fig, ax = plt.subplots(1,1)
    ax.plot(dt_l, x_p_l, "x--")
    ax.plot(dt_l, x_l, "x--")
    ax.plot(dt_l, measurements, "x--")
    ax.plot(dt_l, x_predictions, "x--")
    # ax.scatter(dt_l, x_p_l)
    plt.show()
    print(dt_l)

def example2():
    measurements = [30171,30353,30756,30799,31018,31278,31276,31379,31748,32175]
    a,b = 0.2,0.1
    # a,b = 0.8,0.5
    dt = 5

    # initial conditions:
    x_p = 30000 # initial position
    x_v = 40 # estimated velocity

    # predict:
    x_p = predict2(x_p, x_v, 5)
    x_v = x_v # stays the same
    print(x_p)
    
    # first iteration:
    i=1
    x_p_l = []
    x_predictions = []
    for n in range(len(measurements)):
    # for n in range(2):
        x_p
        x_v
        x_predictions.append(x_p)
        z_i = measurements[i-1]

        x_p_temp = x_p
        x_v_temp = x_v
        x_p = update(x_p_temp, z_i, a)
        x_v = update2(x_v_temp, (z_i-x_p_temp)/dt, b)
        x_p_l.append(x_p)
        
        print("%.2f\n%.2f\n"%(x_p, x_v))
        x_p = x_p + dt*x_v
        x_v = x_v
        i = i+1
    
    dt_l = np.arange(len(measurements))*dt
    x_l = (np.arange(len(measurements))+1)*200+30000
    fig, ax = plt.subplots(1,1)
    ax.plot(dt_l, x_p_l, "x--")
    ax.plot(dt_l, x_l, "x--")
    ax.plot(dt_l, measurements, "x--")
    ax.plot(dt_l, x_predictions, "x--")
    # ax.scatter(dt_l, x_p_l)
    plt.show()
    print(dt_l)


def example1():
    measurements = [996,994,1021,1000,1002,1010,983,971,993,1023]
    x_0 = 1000 # initial guess

    # predict:
    x_p = predict(x_0)

    # first iteration:
    i=1
    for n in range(len(measurements)):
    # for n in range(2):
        z_i = measurements[i-1]
        gain_i = 1/i
        x_i = update(x_p, z_i, gain_i)
        print("%.2f"%x_i)
        x_p = x_i
        i = i+1





if __name__ == '__main__':
    main()