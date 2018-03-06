import mxnet as mx
import numpy as np
import json
import cv2


# get iterator
def get_iterator(path, data_shape, label_width, batch_size, shuffle=False):
    iterator = mx.io.ImageRecordIter(path_imgrec=path,
                                    data_shape=data_shape,
                                    label_width=label_width,
                                    batch_size=batch_size,
                                    shuffle=shuffle)
    return iterator
# idl to numpy (.npy file)
def idltonumpy(idlfile,savepath):
    label_np = {}
    with open(idlFile_path) as f:
        for line in f:
            jsonload = json.loads(line) # type(line): str  type(jsonload):dir
            assert len(jsonload.keys()) ==1, "only one image per json file"
            label_np[list(jsonload.keys())[0]] = jsonload[list(jsonload.keys())[0]]
    np.save(save_path,label_np)
    return label_np

# box:[lelf_up_x,lelf_up_y,lelf_down_x,lelf_dowm_y]
# to box:[center_x,center_y,width,high] in pixel
def xy2wh(idlnpy):
    labelwh = {}
    for key in idlnpy.keys():
        boxxy = idlnpy[key]
        boxeswh = []
        for each in boxxy:
            x1, y1, x2, y2, idx = each
            cx = int((x2+x1)/2)
            cy = int((y1+y2)/2)
            w = int(x2-x1)
            h = int(y2-y1)
            # one-hod encoding
            cls = [0,0,0,0]
            if idx==1:
                cls[0]=1
            elif idx==2:
                cls[1]=1
            elif idx==3:
                cls[2]=1
            elif idx==20:
                cls[3]=1
            else:
                print(idx)
                print("Not expected value")
                break
            assert w>0 and h>0, "wh should be > 0"
            box = [cx, cy, w, h] + cls
            boxeswh.append(box)
        labelwh[key] = boxeswh
    return labelwh

# Convert data to rec file
# Get Yolo x and y
def get_YOLO_xy(bxy, grid_size=(7,7), dscale=32, sizet=224):
    cx, cy = bxy
    assert cx<=1 and cy<=1, "All should be < 1, but get {}, and {}".format(cx,cy)

    j = int(np.floor(cx/(1.0/grid_size[0])))
    i = int(np.floor(cy/(1.0/grid_size[1])))
    xyolo = (cx * sizet - j * dscale) / dscale
    yyolo = (cy * sizet - i * dscale) / dscale
    return [i, j, xyolo, yyolo]

# Get YOLO label
def imgResizeBBoxTransform(img, bbox, sizet, grid_size=(7,7,5), dscale=32):

    himg, wimg = img.shape[:2]
    imgR = cv2.resize(img, dsize=(sizet, sizet))
    bboxyolo = np.zeros(grid_size)
    for eachbox in bbox:
        cx, cy, w, h = eachbox[:4]
        cls = eachbox[4:]
        cxt = 1.0*cx/wimg
        cyt = 1.0*cy/himg
        wt = 1.0*w/wimg
        ht = 1.0*h/himg
        assert wt<1 and ht<1
        i, j, xyolo, yyolo = get_YOLO_xy([cxt, cyt], grid_size, dscale, sizet)
        #print "one yolo box is {}".format((i, j, xyolo, yyolo, wt, ht))
        label_vec = np.asarray([1, xyolo, yyolo, wt, ht]+cls)
        # print(label_vec)
        #print "Final yolo box is {}".format(label_vec)
        bboxyolo[i, j, :] = label_vec
    return imgR, bboxyolo

# Conver raw image to rec files
def toRecFile(imgroot,imglist,annotation,sizet,grid_size,descale,recfile_name):
    recF_name = "DATA_rec/"+recfile_name
    record = mx.recordio.MXIndexedRecordIO(recF_name+".idx",recF_name+".rec",'w')
    for i in range(len(imglist)):
        imgname = imglist[i]
        img = cv2.imread(imgroot+imgname)
        bbox = annotation[imgname]
        print("Now is processing img {}".format(imgname))
        imgR,bboxR = imgResizeBBoxTransform(img,bbox,sizet,grid_size,descale)
        header = mx.recordio.IRHeader(flag=0, label=bboxR.flatten(), id=0, id2=0)
        s = mx.recordio.pack_img(header, imgR, quality=100, img_fmt='.jpg')
        record.write_idx(i,s)
    print("JPG to rec is Done")
    record.close()

if __name__ == "__main__":
    # transform jpg to rec file
    trainDATA_path = "trainDATA/training/"
    idlFile_path = "trainDATA/training/label.idl"
    save_path = "trainDATA/training/label.npy"

    label_np = idltonumpy(idlFile_path, save_path)
    labelwh = xy2wh(label_np)
    imglist = list(label_np.keys())
    sizet = 224
    grid_size = (7, 7, 9)
    descale = 32
    full_recfile_name = "drive_full"
    toRecFile(trainDATA_path, imglist, labelwh, sizet, grid_size, descale,full_recfile_name)