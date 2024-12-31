import torch
import torch.nn.functional as F
from torchsummary import summary
import cv2 as cv
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


import sys
sys.path.append('../')
sys.path.append('../../RAFT/core')
from helper_func import *

import math
import itertools
from functools import partial

class ImagePadder:
    def __init__(self, shape, patch_size=14):
        self.patch_size = patch_size
        
        dimensions = shape
        if(len(dimensions) < 2):
            print("[warn] incorrect input dimensions, should set manually")
            return
        # print(dimensions)
        h, w = dimensions[0], dimensions[1]
        self.h, self.w = h,w

        # print(h//patch_size, w//patch_size )
        h_d = h//patch_size
        w_d = w//patch_size

        # get residual
        h_res = h-h_d*patch_size
        w_res = w-w_d*patch_size

        w_pad = patch_size - w_res
        h_pad = patch_size - h_res
        if(h_res == 0):
            h_pad = 0
        if(w_res == 0):
            w_pad = 0
        # print("pad vals: ", h_pad, w_pad)

        pad_l = w_pad//2
        pad_r = w_pad - pad_l

        pad_u = h_pad//2
        pad_d = h_pad - pad_u

        # save dimensions
        self.pad_l = pad_l
        self.pad_r = pad_r
        self.pad_u = pad_u
        self.pad_d = pad_d

        # save also patch dimensions (how many patches fit in h, w)
        self.patch_h = (h+h_pad)//patch_size
        self.patch_w = (w+w_pad)//patch_size

        print("[padder] ",h, w," -> ", (h+h_pad), (w+w_pad))
    
    def im_pad(self, im, mode='reflect'):
        # return np.pad(im, pad_width=[(self.pad_u,self.pad_d),(self.pad_l,self.pad_r),(0,0)]) # h, w, rgb
        # return np.pad(im, pad_width=[(self.pad_u,self.pad_d),(self.pad_l,self.pad_r),(0,0)], mode='reflect') # h, w, rgb
        return np.pad(im, pad_width=[(self.pad_u,self.pad_d),(self.pad_l,self.pad_r),(0,0)], mode=mode) # h, w, rgb
    
    def im_pad2d(self, im, mode='reflect'):
        return np.pad(im, pad_width=[(self.pad_u,self.pad_d),(self.pad_l,self.pad_r)], mode=mode) # h, w, rgb
    
    def im_unpad(self, im):
        return im[self.pad_u: (self.h-self.pad_d), self.pad_l: (self.w-self.pad_r)]


DEVICE = 'cuda'
BACKBONE_SIZE = "small" # in ("small", "base", "large" or "giant")

DATA_ROOT = '../../data/'
ESTIMATES_ROOT = 'estimates/'
DEVICE = 'cuda'
CALCULATE_FLOW = True

# pingvini5
ESTIMATES_FILE = 'sample_flo_2.p'
DETS_FILE = 'sample/dets.txt'
FRAMES_PATH = 'sample/'

def load_image(imfile):
    img = cv.imread(imfile).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def permute201(image, device='cpu'):
    img = torch.from_numpy(image).permute(2, 0, 1).float()
    return img[None].to(device)

def permute120(image, device='cpu'):
    img = image.permute(1,2,0).to(device).numpy()
    return img


def getFeatures():
    print("this is get_features")
    
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    # dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    backbone_model.to(DEVICE)
    backbone_model.eval()

    # print(backbone_model)
    # summary(backbone_model, input_size=(3,224,224), batch_size=1)

    # show image
    image = getFrameAtI(0, DATA_ROOT+FRAMES_PATH)
    # return 
    winname = "image"
    win_feat = "features"
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.imshow(winname, image)
    cv.waitKey(0)

    # TODO: get features (test it)
    with torch.no_grad():
        ip = ImagePadder(np.shape(image))
        image2 = ip.im_pad(image)
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
        features = backbone_model(torch.from_numpy( np.array([image2,image2])).to('cuda'))
        # print(features.feature_maps)
        cv.namedWindow(win_feat, cv.WINDOW_NORMAL)
        print(features)
        # cv.imshow(win_feat, features)
        # cv.waitKey(0)

class DinoFeatures():
    def __init__(self, model_size=BACKBONE_SIZE):
        # init model
        print("[INIT MODEL]")
        backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
        }
        backbone_arch = backbone_archs[model_size]
        backbone_name = f"dinov2_{backbone_arch}"

        # init backbone model
        self.model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
        self.model.to(DEVICE)
        self.model.eval()

        # save last padder for image dimension retrieval
        self.last_padder: ImagePadder = None

        print("model is: ")
        print(self.model)
        print("[END INIT MODEL]")
        # summary(backbone_model, input_size=(3,224,224), batch_size=1)
    
    def prepareImageForModel(self, image: np.ndarray, to_device=DEVICE) -> torch.Tensor:
        # 1. pad it
        # 2. convert it to torch.Tensor and move it to cuda (also permute dimensions)
        # 3. return image
        
        
        
        ip = ImagePadder(np.shape(image))
        self.last_padder = ip
        image2 = np.array([ip.im_pad(image)])
        
        image2 = torch.from_numpy(image2).permute(0,3, 1, 2).float()
        if(to_device == 'cuda'):
            image2 = image2.to(to_device)
        return image2
    
    def getFeatures(self, image: np.ndarray, to_device=DEVICE):
        with torch.no_grad():
            image2 = self.prepareImageForModel(image, to_device=None)
            # print(image2)
            # p_features: torch.Tensor = self.model(image2.to(to_device))
            # print(p_features)
            feature_dict: torch.Tensor = self.model.forward_features(image2.to(to_device))
            # print(feature_dict)
            # if(to_device == 'cuda'):
            #     features = features.to('cpu').numpy() # convert it to cpu
            return feature_dict
    
    def cosineSim(self, f1, f2):
        return np.dot(f1,f2.T) / (np.sqrt(np.dot(f1,f1.T))*np.sqrt(np.dot(f2,f2.T)))[0][0]
    
    def l2(self, f1, f2):
        diff = f1-f2
        return np.sqrt(np.dot(diff,diff.T))[0][0]

def showInNamed(name, image):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, image)

def patchesInBB(patches_img, bb, xy_off=[0,0], patch_size:int = 14):
    # 1. get og. image size:
    # patches_h, patches_w, _ = np.shape(patches_img)
    # og_h, og_w = patches_h*patch_size, patches_w*patch_size
    # xy_off: we need to correct x and y values due to padding of image (basically we need to add left and upper padding to x and y, respectively)
    patch_ul_x, patch_ul_y = (bb[0]+xy_off[0])//patch_size, (bb[1]+xy_off[1])//patch_size
    
    patch_bb_h = bb[2]//patch_size + (1 if(int(bb[2])%patch_size != 0) else 0)
    patch_bb_w = bb[3]//patch_size + (1 if(int(bb[3])%patch_size != 0) else 0)
    
    
    print("patches in bb, x: ",bb[0],patch_ul_x, patch_bb_h)
    print("patches in bb, y: ",bb[1],patch_ul_y, patch_bb_w)

    return patches_img[patch_ul_y:patch_ul_y+patch_bb_w, patch_ul_x:patch_ul_x+patch_bb_h]
    # return patches_img[patch_ul_x:patch_ul_x+patch_bb_h, patch_ul_y:patch_ul_y+patch_bb_w]
    

# n: array
def roiPool(mat: np.ndarray, r:int = 3, use_2d=False) -> np.ndarray:
    ip = ImagePadder(np.shape(mat)[0:2], patch_size=r)

    mat_p = mat
    if(use_2d):
        mat_p = ip.im_pad2d(mat,mode='edge')
    else:
        mat_p = ip.im_pad(mat,mode='edge')
    # print("mat_p:",mat_p)
    showInNamed("mat_p", mat_p[..., ::-1])
    pshape = np.shape(mat_p)
    h, w = pshape[0:2]

    d_h = h//r
    d_w = w//r
    roi_pooled = np.zeros((r,r)).tolist() # !!!!! data type might be different
    
    for i in range(r):
        for j in range(r):
            mat_p_ij = mat_p[ i*d_h: (i+1)*d_h, j*d_w: (j+1)*d_w]
            # print("mat_pij:", mat_p_ij)
            # print("mat_pij:",mat_p_ij)
            pooled_ij = np.max(mat_p_ij, axis=0)[0] # !! [TODO - ne deluje se cist vredu, treba je drugo funkcijo]
            roi_pooled[i][j] = pooled_ij
    return np.array(roi_pooled)

def cosineSim(v1,v2):
    v1 = np.array(v1).astype(np.float32)
    v2 = np.array(v2).astype(np.float32)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    v1dotv2 = np.dot(v1,v2)
    score = v1dotv2/(v1_norm*v2_norm)
    # print("v1, v1 norm, v2, v2 norm, v1 dot v2, cosSim:", v1, v2, v1_norm, v2_norm, v1dotv2, score)
    return score


def matSimScore(m1, m2):
    # assume matrices are of same shape (3x3, (roiPooled features))
    print("m1,m2:",m1, m2)
    hw_ = np.shape(m1)
    h, w = hw_[0:2]
    score_mat = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            # compute cosine similarity
            v1 = m1[i,j]
            v2 = m2[i,j]
            cos_sim_score = cosineSim(v1,v2)
            score_mat[i,j] = cos_sim_score
            
    
    # if cosine similarity: set neg values to 0
    print("score mat:",score_mat)
    score_mat[np.where(score_mat < 0)] = 0
    return np.mean(score_mat)


# function contains code with functionality that of final implementation of getting visual features of roi-s from frames
def main2():
    a = np.array([125,136,182])
    b = np.array([125,136,182])
    print(cosineSim(a,b))
    
    a = np.array([1,0])
    b = np.array([3,0])
    print(cosineSim(a,b))
    
    # # a = np.array([1,0])
    # # b = np.array([0,2])
    # # print(cosineSim(a,b))
    
    # return
    # init model:
    dinov2 = DinoFeatures()
    
    # 1. get image
    image = getFrameAtI(0, DATA_ROOT+FRAMES_PATH)
    
    
    # 2. get image features:
    features = dinov2.getFeatures(image=image)["x_norm_patchtokens"].to('cpu').numpy()
    patch_h, patch_w = dinov2.last_padder.patch_h, dinov2.last_padder.patch_w
    pad_l, pad_u = dinov2.last_padder.pad_l, dinov2.last_padder.pad_u
    print("featurji:")
    print(features)

    # 3. PCA: to get rid of background patches:
    # PCA
    thresh = 0
    
    
    features1 = features.reshape(patch_h*patch_w, 384) # tile image with patches (for each patch we have feature vector)
    print(np.shape(features1))

    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features1)
    # pca_features = pca.transform(features1)

    print(np.shape(pca_features))
    print(pca_features)
    
    # segment using the first component (background, foreground:)
    pca_features_bg = pca_features[:, 0] < thresh # if first component is negative, then it is bg
    pca_features_fg = ~pca_features_bg

    
    # 4. get regions of interest (bounding boxes)
    # d1:
    bb1 = (264,202,27,64) # p1
    im_bb1 = getRectBb(bb1,image)

    # d2:
    bb2 = (199,238,38,83) # p2
    im_bb2 = getRectBb(bb2,image)
    
    # d3:
    bb3 = (340,184,30,70) # not p
    im_bb3 = getRectBb(bb3,image)

    showInNamed("p1", im_bb1)
    showInNamed("p2", im_bb2)
    showInNamed("p3", im_bb3)


    
    # visualization: (pca 3 (for rgb, thats why 3))
    # PCA for only foreground patches (second pca)
    pca.fit(features1[pca_features_fg])
    pca_features_rem = pca.transform(features1[pca_features_fg])
    
    for i in range(3):
        pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].min()) / (pca_features_rem[:, i].max() - pca_features_rem[:, i].min())
        # transform using mean and std, I personally found this transformation gives a better visualization <- visualization!!
        # pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].mean()) / (pca_features_rem[:, i].std() ** 2) + 0.5 # center it

    rem_min = pca_features_rem.min()
    rem_max = pca_features_rem.max()
    pca_features_rem = pca_features_rem - rem_min
    pca_features_rem = (pca_features_rem/(rem_min+rem_max))*255

    pca_features_rgb = pca_features.copy()
    pca_features_rgb[pca_features_bg] = 0 # set background features to 0
    pca_features_rgb[pca_features_fg] = pca_features_rem # first 3 components
    pca_features_rgb = pca_features_rgb.reshape(patch_h, patch_w, 3).astype(np.uint8)



    # patches1 = patchesInBB(pca_features_rgb, bb1, xy_off=[pad_l, pad_u])
    patches1 = patchesInBB(features1[pca_features_fg], bb1, xy_off=[pad_l, pad_u])
    print(patches1)
    # showInNamed("p1 - patches:", patches1[..., ::-1])

    # 5. RoiPooled patches:
    patches_rp1 = roiPool(patches1)
    print(patches_rp1)
    # showInNamed("p1 - pooled (3x3):", patches_rp1[..., ::-1])
    print(matSimScore(patches_rp1, patches_rp1))



    # plt.subplot(1, 1, 1)
    # plt.imshow(pca_features_rgb[..., ::-1])
    showInNamed("features", pca_features_rgb[..., ::-1])
    
    showInNamed("image", image)
    # plt.show()
    # plt.close()
    
    
    cv.waitKey(0)



    
    
def main_d():
    a = np.array([[[1],[2],[3],[4],[5],[6],[7]],[[8],[9],[10],[11],[12],[13],[14]]])
    print(a)
    print(roiPool(a))


def main():
    # init model:
    dinov2 = DinoFeatures()


    image = getFrameAtI(0, DATA_ROOT+FRAMES_PATH)
    
    # p1 = "p1"
    bb1 = (264,202,27,64) # p1
    im_bb1 = getRectBb(bb1,image)

    # p2 = "p2"
    bb2 = (199,238,38,83) # p2
    # im_bb2 = getRectBb(bb2,image)

    # p3 = "p3"
    bb3 = (340,184,30,70) # not p
    # im_bb3 = getRectBb(bb3,image)

    # visualize results:
    winname = "image"
    
    # showInNamed(p1, im_bb1)
    # showInNamed(p2, im_bb2)
    # showInNamed(p3, im_bb3)


    # get features:
    feature_dict1 = dinov2.getFeatures(im_bb1)
    # feature_dict1 = dinov2.getFeatures(image)
    print(feature_dict1)
    feature_vec1 = feature_dict1['x_norm_clstoken']
    features1 = feature_dict1['x_norm_patchtokens'].to('cpu').numpy()
    print(np.shape(features1))
    print(np.mean(features1, axis=1)) # ni isto
    # print(np.mean(features1, axis=0))

    # # get feature vectors:
    # fp1 = dinov2.getFeatures(im_bb1)
    # fp2 = dinov2.getFeatures(im_bb2)
    # fp3 = dinov2.getFeatures(im_bb3)

    drawBoundingBox(image, bb1, [0,255,255])
    drawBoundingBox(image, bb2, [0,255,255])
    drawBoundingBox(image, bb3, [0,255,255])
    showInNamed(winname, image)


    # # print("features of p1:")
    # # print(fp1)
    # # print("features of p2:")
    # # print(fp2)
    # # print("features of p3:")
    # # print(fp3)

    # print("similarities: ")
    # print("p1, p2: ", dinov2.cosineSim(fp1,fp2),dinov2.l2(fp1,fp2))
    # print("p1, p3: ", dinov2.cosineSim(fp1,fp3),dinov2.l2(fp1,fp3))
    # print("p2, p3: ", dinov2.cosineSim(fp2,fp3),dinov2.l2(fp2,fp3))
    # print("p1, p1: ", dinov2.cosineSim(fp1,fp1),dinov2.l2(fp1,fp1))

    # PCA
    patch_h = 26
    patch_w = 35

    patch_h = 5
    patch_w = 2
    # thresh = 40
    thresh = 38
    thresh = 5
    thresh = 0.9
    # im_h, im_w, _ = np.shape(im_bb1)
    features1 = features1.reshape(patch_h*patch_w, 384)
    print(np.shape(features1))

    pca = PCA(n_components=3)
    pca.fit(features1)
    pca_features = pca.transform(features1)

    print(np.shape(pca_features))
    print(pca_features)
    

    # visualize PCA components for finding a proper threshold
    plt.subplot(1, 3, 1)
    plt.hist(pca_features[:, 0])
    plt.subplot(1, 3, 2)
    plt.hist(pca_features[:, 1])
    plt.subplot(1, 3, 3)
    plt.hist(pca_features[:, 2])
    plt.show()
    plt.close()


    # segment using the first component
    pca_features_bg = pca_features[:, 0] < 0 # if first component is negative, then it is bg
    pca_features_fg = ~pca_features_bg

    print(pca_features_bg)

    # plot the pca_features_bg
    for i in range(1):
        plt.subplot(2, 2, i+1)
        plt.imshow(pca_features_bg[i * patch_h * patch_w: (i+1) * patch_h * patch_w].reshape(patch_h, patch_w))
    plt.show()

    # PCA for only foreground patches (second pca)
    pca.fit(features1[pca_features_fg]) # NOTE: I forgot to add it in my original answer
    pca_features_rem = pca.transform(features1[pca_features_fg])
    print("pca features rem:")
    print(pca_features_rem)

    for i in range(3):
        # pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].min()) / (pca_features_rem[:, i].max() - pca_features_rem[:, i].min())
        # transform using mean and std, I personally found this transformation gives a better visualization <- visualization!!
        pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].mean()) / (pca_features_rem[:, i].std() ** 2) + 0.5 # center it

    pca_features_rgb = pca_features.copy()
    pca_features_rgb[pca_features_bg] = 0 # set background features to 0
    pca_features_rgb[pca_features_fg] = pca_features_rem # first 3 components

    pca_features_rgb = pca_features_rgb.reshape(1, patch_h, patch_w, 3)

    print(pca_features_rgb)
    print("pooled features:")
    print(np.shape(pca_features_rgb[0]))
    print()
    print(roiPool(pca_features_rgb[0]))

    print("pca features rgb")
    print(pca_features_rgb[0])
    print(pca_features_rgb[0][..., ::-1])
    print("-----------------")

    for i in range(1):
        plt.subplot(2, 2, i+1)
        plt.imshow(pca_features_rgb[i][..., ::-1])
        # plt.imshow(pca_features_rgb[i])
    plt.savefig('features.png')
    plt.show()
    plt.close()
    
    cv.waitKey(0)


if __name__ == "__main__":
    # main()
    # main_d()
    main2()