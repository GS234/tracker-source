import torch
import cv2 as cv
from sklearn.decomposition import PCA
from helper_func import *

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

        # print("[padder] ",h, w," -> ", (h+h_pad), (w+w_pad))
    
    def im_pad(self, im, mode='reflect'):
        # return np.pad(im, pad_width=[(self.pad_u,self.pad_d),(self.pad_l,self.pad_r),(0,0)]) # h, w, rgb
        # return np.pad(im, pad_width=[(self.pad_u,self.pad_d),(self.pad_l,self.pad_r),(0,0)], mode='reflect') # h, w, rgb
        return np.pad(im, pad_width=[(self.pad_u,self.pad_d),(self.pad_l,self.pad_r),(0,0)], mode=mode) # h, w, rgb
    
    def im_pad2d(self, im, mode='reflect'):
        return np.pad(im, pad_width=[(self.pad_u,self.pad_d),(self.pad_l,self.pad_r)], mode=mode) # h, w, rgb
    
    def im_unpad(self, im):
        return im[self.pad_u: (self.h-self.pad_d), self.pad_l: (self.w-self.pad_r)]

def getImagePadding(im_shape, patch_size=14):
    h, w = im_shape[0], im_shape[1]
    
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

    pad_l = w_pad//2
    pad_r = w_pad - pad_l

    pad_u = h_pad//2
    pad_d = h_pad - pad_u

    # return paddings
    return (pad_l, pad_r, pad_u, pad_d)


DEVICE = 'cuda'
# BACKBONE_SIZE = "small" # in ("small", "base", "large" or "giant")
# BACKBONE_SIZE = "base"
BACKBONE_SIZE = "base"


class FeatureExtractor():
    def __init__(self, model_size=BACKBONE_SIZE, patch_size=14, no_print=False):
        # init model
        print("[FEATURE] init model")
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[model_size]
        backbone_name = f"dinov2_{backbone_arch}"

        # init backbone model
        # self.model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name, patch_size=patch_size)
        # self.model = torch.hub.load(repo_or_dir="/home/gasper/python/dinov2/facebookresearch_dinov2_main/", source='local', trust_repo=True, model=backbone_name, patch_size=patch_size)
        self.model = torch.hub.load(repo_or_dir="/home/gasper/.cache/torch/hub/facebookresearch_dinov2_main/", source='local', trust_repo=True, model=backbone_name, patch_size=patch_size)
        self.model.to(DEVICE)
        self.model.eval()

        # save last padder for image dimension retrieval
        self.last_padder: ImagePadder = None

        # print("model is: ")
        # print(self.model)
        print("[FEATURE] init done")
        self.no_print = no_print
        # summary(backbone_model, input_size=(3,224,224), batch_size=1)
    
    def prepareImageForModel(self, image: np.ndarray, to_device=DEVICE) -> torch.Tensor:
        # 1. pad it
        # 2. convert it to torch.Tensor and move it to cuda (also permute dimensions)
        # 3. return image
        
        ip = ImagePadder(np.shape(image))
        self.last_padder = ip
        image2 = np.array([ip.im_pad(image)])
        
        image2 = torch.from_numpy(image2).permute(0,3, 1, 2).float()
        # image is on cpu, move it to cuda, if needed
        if(to_device != 'cpu'):
            image2 = image2.to(to_device)
        return image2
    
    def getFeatures(self, image: np.ndarray, to_device=DEVICE):
        with torch.no_grad():
            image2 = self.prepareImageForModel(image, to_device=to_device)
            # print(image2)
            # p_features: torch.Tensor = self.model(image2.to(to_device))
            # print(p_features)
            feature_dict: torch.Tensor = self.model.forward_features(image2.to(to_device))
            # feature_dict: torch.Tensor = self.model.forward_features(image2)
            # print(feature_dict)
            # if(to_device == 'cuda'):
            #     features = features.to('cpu').numpy() # convert it to cpu
            # print(feature_dict)
            return feature_dict
    
    def l2(self, f1, f2):
        diff = f1-f2
        return np.sqrt(np.dot(diff,diff.T))[0][0]


def patchesInBB(patches_img, bb, xy_off=[0,0], patch_size:int = 14):
    # 1. get og. image size:
    # patches_h, patches_w, _ = np.shape(patches_img)
    # og_h, og_w = patches_h*patch_size, patches_w*patch_size
    # xy_off: we need to correct x and y values due to padding of image (basically we need to add left and upper padding to x and y, respectively)
    # print(bb)
    bb = np.array(bb)
    bb = np.where(bb < 0, 0, bb)
    bb = bb.astype(np.uint32)
    # print("bb after:")
    # print(bb)
    patch_ul_x, patch_ul_y = (bb[0]+xy_off[0])//patch_size, (bb[1]+xy_off[1])//patch_size
    
    patch_bb_h = bb[2]//patch_size + (1 if(int(bb[2])%patch_size != 0) else 0)
    patch_bb_w = bb[3]//patch_size + (1 if(int(bb[3])%patch_size != 0) else 0)
    
    
    # print("patches in bb, x: ",bb[0],patch_ul_x, patch_bb_h)
    # print("patches in bb, y: ",bb[1],patch_ul_y, patch_bb_w)

    return patches_img[patch_ul_y:patch_ul_y+patch_bb_w, patch_ul_x:patch_ul_x+patch_bb_h]
    # return patches_img[patch_ul_x:patch_ul_x+patch_bb_h, patch_ul_y:patch_ul_y+patch_bb_w]
    

# n: array
def roiPool(mat: np.ndarray, r:int = 3, use_2d=False, debug=False) -> np.ndarray:
    # print(np.shape(mat))
    mat_shape = np.shape(mat)[0:2]
    # print(mat_shape)
    if(0 in mat_shape):
        print("[WARN] (roiPool) size of one edge of mat is 0 (%s), returning zeros"%(str(mat_shape)))
        return np.zeros((r,r,3))

    
    ip = ImagePadder(np.shape(mat)[0:2], patch_size=r)

    mat_p = mat
    if(use_2d):
        mat_p = ip.im_pad2d(mat,mode='edge')
    else:
        mat_p = ip.im_pad(mat,mode='edge')
    
    # print("mat_p:",mat_p)
    # showInNamed("mat_p", mat_p[..., ::-1])
    # showInNamed("abc", mat_p[..., ::-1])
    
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
            pooled_ij = np.max(mat_p_ij.reshape((-1,3)), axis=0) # max pool
            # pooled_ij = np.max(mat_p_ij, axis=0)[0] # max pool
            roi_pooled[i][j] = pooled_ij
    # return np.array(roi_pooled)[..., ::-1]
    return np.array(roi_pooled)

def getFgMask(features):
    # PCA: to get rid of background patches:
    thresh = 0
    features1 = features[0]

    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features1)
    # print("pca features")
    # print(np.shape(pca_features))
    
    # segment using the first component (background, foreground:)
    # MASKS!:
    pca_features_bg = pca_features[:, 0] < thresh # if first component is negative, then it is bg
    pca_features_fg = ~pca_features_bg
    return pca_features_fg

def getFeaturesPca(features, patch_hw: tuple[int, int]):
    pca_features_fg = getFgMask(features)
    n_components = 3
    features1 = features[0]
    pca = PCA(n_components=n_components)
    pca_features_rem = pca.fit_transform(features1[pca_features_fg])
    # pca_features_rem = pca.fit_transform(features1)
    # pca_features_rem = pca.transform(features1[pca_features_fg])
    
    for i in range(n_components):
        pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].min()) / (pca_features_rem[:, i].max() - pca_features_rem[:, i].min())
        # transform using mean and std, I personally found this transformation gives a better visualization <- visualization!!
        # pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].mean()) / (pca_features_rem[:, i].std() ** 2) + 0.5 # center it

    rem_min = pca_features_rem.min()
    rem_max = pca_features_rem.max()
    pca_features_rem = pca_features_rem - rem_min
    pca_features_rem = (pca_features_rem/(rem_min+rem_max))*255

    feat_shape = (np.shape(features1)[0],n_components)
    
    # print("feat_shape:")
    # print(feat_shape)
    pca_features_rgb = np.zeros(feat_shape)
    pca_features_rgb[~pca_features_fg] = 0 # set background features to 0
    pca_features_rgb[pca_features_fg] = pca_features_rem # first 3 components
    # pca_features_rgb = pca_features_rem
    patch_h, patch_w = patch_hw
    # print(np.shape(pca_features_rgb))
    pca_features_rgb = pca_features_rgb.reshape(patch_h, patch_w, n_components).astype(np.uint8)
    return pca_features_rgb


def main3():
    image_path = '/home/gasper/Desktop/' # muc

    dinov2 = FeatureExtractor()
    
    # 1. get image
    image = getFrameAtI(0, image_path)
    
    
    # 2. get image features:
    features = dinov2.getFeatures(image=image)["x_norm_patchtokens"].to('cpu').numpy() # patch tokens
    print(np.shape(features))
    patch_h, patch_w = dinov2.last_padder.patch_h, dinov2.last_padder.patch_w
    pad_l, pad_u = dinov2.last_padder.pad_l, dinov2.last_padder.pad_u
    print("featurji:")
    print(features)

    pca_features_rgb = getFeaturesPca(features, (patch_h, patch_w))


    showInNamed("features", pca_features_rgb)
    showInNamed("image", image)
    
    
    cv.waitKey(0)

    
    pass


# quick test:
def main2():
    FRAMES_PATH = '/home/gasper/disk/Nedokumenti/Faks/didi_sequences/LaSOT_bird-2/color/'
    # init model:
    dinov2 = FeatureExtractor()
    
    # 1. get image
    image = getFrameAtI(1, FRAMES_PATH)
    
    
    # 2. get image features:
    features = dinov2.getFeatures(image=image)["x_norm_patchtokens"].to('cpu').numpy() # patch tokens
    print(np.shape(features))
    patch_h, patch_w = dinov2.last_padder.patch_h, dinov2.last_padder.patch_w
    pad_l, pad_u = dinov2.last_padder.pad_l, dinov2.last_padder.pad_u
    print("featurji:")
    print(features)

    # 3. PCA: to get rid of background patches:
    pca_features_fg = getFgMask(features) # does pca

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

    pca_features_rgb = getFeaturesPca(features, (patch_h, patch_w))

    patches2 = patchesInBB(pca_features_rgb, bb1, xy_off=[pad_l, pad_u])
    print("patches2:")
    print(patches2)

    # 5. RoiPooled patches:
    patches_rp1 = roiPool(patches2, use_2d=False)
    print(patches_rp1)
    showInNamed("p1 - pooled (3x3):", patches_rp1)

    showInNamed("features", pca_features_rgb)
    showInNamed("image", image)
    
    
    cv.waitKey(0)

if __name__ == '__main__':
    main3()

