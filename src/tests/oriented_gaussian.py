import numpy as np
import cv2 as cv

def gaussian_base(x, t=0.2):
    return np.exp(-t*(x*x))

# generates bounding points
def search_region(s1, s2, c=None, phi=0, n=20):
    if c is None:
        c = [s1,s2]
    rot_mat = np.array([[np.cos(phi), -np.sin(phi)],[np.sin(phi), np.cos(phi)]]) # rotation matrix
    print(rot_mat)
    full_circle = np.arange(0, 2*np.pi, 2*np.pi/n)
    
    X = np.array([s1*np.sin(full_circle), s2*np.cos(full_circle)])
    X_rot = np.dot(rot_mat, X)
    # X_rot = np.dot(rot_mat.T, X_rot) # rotate back same amount (inverse rotation)
    X_rot[0] += c[0]
    X_rot[1] += c[1]

    return X_rot

def vector_in_region(x, s1, s2, c=None, phi=0, n=20):
    if c is None:
        c = [s1,s2]
    # also show vector: rotated and not rotated
    rot_mat = np.array([[np.cos(phi), -np.sin(phi)],[np.sin(phi), np.cos(phi)]]) # rotation matrix
    x_rot = np.dot(rot_mat, x)
    search_reg = search_region(s1, s2, c, phi, n)

    

    x_rot[0] += c[0]
    x_rot[1] += c[1]
    x[0] += c[0]
    x[1] += c[1]

    x = x.reshape((1, len(x)))
    x_rot = x_rot.reshape((1, len(x_rot)))
    
    return np.hstack((search_reg, x.T, x_rot.T))



# main:
def main():
    # n = 10
    # # print(gaussian_base(10))
    # zaporedje = np.arange(-n//2,n//2+1,1)
    # gaussian_x = gaussian_base(zaporedje)
    # gaussian_x = gaussian_x.reshape((1,len(gaussian_x)))
    # gaussian_y = gaussian_base(zaporedje, 0.1)
    # gaussian_y = gaussian_y.reshape((1,len(gaussian_y)))
    # c = np.zeros((n+1,n+1))
    # c[n//2,n//2]=1.0
    
    # c=cv.filter2D(c, ddepth=-1, kernel=gaussian_x)
    # c=cv.filter2D(c, ddepth=-1, kernel=gaussian_y.T)


    # cv.imshow("a",c)
    # cv.waitKey(0)
    n = 500
    some_vec = np.array([0,5])
    # X = search_region(10,20, [n//2,n//2], (np.pi/180)*30, n=100)
    X = vector_in_region(some_vec,10,20, [n//2,n//2], (np.pi/180)*30, n=100)

    X = X.astype(np.int32)
    # print(X)
    c = np.zeros((n,n))

    
    

    for x in X.T:
        c[x[0],x[1]] = 1.0

    cv.imshow("abc", c)
    cv.waitKey(0)














if __name__ == "__main__":
    main()