import mxnet as mx


def expit_tensor(x):
    return 1/(1+mx.sym.exp(-x))

# Yolo Loss

def Yolo_loss(predict,label):
    """
    predict (params): mx.sym->which is NDarray (tensor), its shape is (batch_size, 7, 7,9 )
    label: same as predict   # label value: [0,1]
    """
    # Reshape input (7,7,9) to desired shape
    predict = mx.sym.reshape(predict,shape = (-1,49,9))
    # shift everything to (0,1)
    predict_shift = (predict+1)/2
    label = mx.sym.reshape(label,shape = (-1,49,9))
    # split the tensor in the order of [prob,x,y,w,h,cls_1,cls_2,cls_3,cls_4]
    cl,xl,yl,wl,hl,clsl1,clsl2,clsl3,clsl4 = mx.sym.split(label, num_outputs=9, axis=2)
    cp,xp,yp,wp,hp,clsp1,clsp2,clsp3,clsp4 = mx.sym.split(predict_shift, num_outputs=9, axis=2)
    # clsesl = mx.sym.Concat(clsl1, clsl2, clsl3, clsl4, dim=2)
    # clsesp = mx.sym.Concat(clsp1, clsp2, clsp3, clsp4, dim=2)
    # weight different target differently
    lambda_coord = 5
    lambda_obj = 1
    lambda_noobj = 0.2
    #  the number grid of obj << the number grid of no_obj
    mask = cl * lambda_obj + (1 - cl) * lambda_noobj   # mask (-1,49)

    # linear regression
    lossc = mx.sym.LinearRegressionOutput(label=cl*mask, data=cp*mask)
    lossx = mx.sym.LinearRegressionOutput(label=xl*cl*lambda_coord, data=xp*cl*lambda_coord)
    lossy = mx.sym.LinearRegressionOutput(label=yl*cl*lambda_coord, data=yp*cl*lambda_coord)
    lossw = mx.sym.LinearRegressionOutput(label=mx.sym.sqrt(wl)*cl*lambda_coord,
                                          data=mx.sym.sqrt(wp)*cl*lambda_coord)
    lossh = mx.sym.LinearRegressionOutput(label=mx.sym.sqrt(hl)*cl*lambda_coord,
                                          data=mx.sym.sqrt(hp)*cl*lambda_coord)
    losscls1 = mx.sym.LinearRegressionOutput(label=clsl1*cl, data=clsp1*cl)
    losscls2 = mx.sym.LinearRegressionOutput(label=clsl2*cl, data=clsp2*cl)
    losscls3 = mx.sym.LinearRegressionOutput(label=clsl3*cl, data=clsp3*cl)
    losscls4 = mx.sym.LinearRegressionOutput(label=clsl4*cl, data=clsp4*cl)
    losscls = losscls1 + losscls2 + losscls3 + losscls4
    #return joint loss
    loss = lossc + lossx + lossy + lossw + lossh + losscls
    return loss

# Get pretrined imagenet model
def get_resnet_model_YoloV1(model_path,epoch):
    # not necessary to be this name, you can do better
    label = mx.sym.Variable('softmax_label')
    # load symbol and actual weight
    sym,args,aux = mx.model.load_checkpoint(model_path,epoch)
    # extran last bn layer (batch norm)
    sym =sym.get_internals()['bn1_output']
    # append two layers
    sym = mx.sym.Activation(data=sym,act_type="relu",name="relu_final")
    sym = mx.sym.Convolution(data=sym,kernel=(3,3),num_filter=9,
                             pad=(1,1),stride=(1,1),no_bias=True
                             )
    # get softsign [-1,1]
    sym = sym/(1+mx.sym.abs(sym))
    # (-1, 9, 7, 7)-->(-1, 7, 7, 9(c,x,y,w,h))
    logit = mx.sym.transpose(sym, axes=(0, 2, 3, 1), name="logit")
    # apply loss
    loss_ = Yolo_loss(logit, label)
    # mxnet special requirement
    loss = mx.sym.MakeLoss(loss_)
    # multi-output logit should be blocked from generating gradients
    out = mx.sym.Group([loss, mx.sym.BlockGrad(logit)])
    return out

