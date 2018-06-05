import os, sys, shutil, time
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.chairs
import tflib.plot
import tflib.objs.gan_inference
import tflib.utils.distance


DATA_DIR = './dataset/chairs'
'''
hyperparameters
'''
# model type
MODE = 'local_ep' # local_ep, local_epce-z, ali, alice-z
POS_MODE = 'naive_mean_field' # gsp, naive_mean_field, inverse
ALI_MODE = 'concat_x' # concat_x, concat_z, 3dcnn
OP_COM_MODE = 'concat' # concat
OP_DYN_MODE = 'res_w' # res
BN_FLAG = False
BN_FLAG_G = BN_FLAG # batch norm in G
BN_FLAG_E = BN_FLAG # batch norm in E
BN_FLAG_D = BN_FLAG # batch norm in E
BN_FLAG_OP = False # batch norm in operator
# model size
DIM_LATENT_G = 128 # global latent variable
DIM_LATENT_L = 8 # local latent variable
DIM_LATENT_T = DIM_LATENT_L # transformation latent variable
DIM = 32 # model size of frame generator
DIM_OP = 256 # model size of the dynamic operator
# data
LEN = 31 # data length
OUTPUT_SHAPE = [3, 64, 64] # data shape
OUTPUT_DIM = np.prod(OUTPUT_SHAPE) # data dim
# optimization
LAMBDA = 0.1 # reconstruction
LR = 1e-4 # learning rate
BATCH_SIZE = 50 # batch size
BETA1 = .5 # adam
BETA2 = .999 # adam
ITERS = 40000 # number of iterations to train
CRITIC_ITERS = 1
# visualization
N_VIS = BATCH_SIZE


'''
logs
'''
filename_script=os.path.basename(os.path.realpath(__file__))
outf=os.path.join("result", os.path.splitext(filename_script)[0])
outf+='.MODE-'
outf+=str(MODE)
outf+='.ALI_MODE-'
outf+=str(ALI_MODE)
outf+='.DIM-'
outf+=str(DIM)
outf+='.OP_DYN_MODE-'
outf+=str(OP_DYN_MODE)
outf+='.LEN-'
outf+=str(LEN)
outf+='.'
outf+=str(int(time.time()))
if not os.path.exists(outf):
    os.makedirs(outf)
logfile=os.path.join(outf, 'logfile.txt')
shutil.copy(os.path.realpath(__file__), os.path.join(outf, filename_script))
lib.print_model_settings_to_file(locals().copy(), logfile)

ratio = [1.0,]*(LEN-1) + [1, LEN]
ratio = np.asarray(ratio) * 1.0 / (len(ratio) + LEN - 1)
print ratio


'''
models
'''
def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ImplicitOperator(z_l, epsilon, name):
    output = tf.concat([z_l, epsilon], axis=1)
    output = lib.ops.linear.Linear(name+'.Input', DIM_LATENT_L+DIM_LATENT_T, DIM_OP, output)
    output = LeakyReLU(output)

    output = lib.ops.linear.Linear(name+'.1', DIM_OP, DIM_OP, output)
    output = LeakyReLU(output)
    
    output = lib.ops.linear.Linear(name+'.Output', DIM_OP, DIM_LATENT_L, output)

    if OP_DYN_MODE == 'res':
        output = output + z_l
    
    elif OP_DYN_MODE == 'res_w':
        output = output + lib.ops.linear.Linear(name+'.ZW', DIM_LATENT_L, DIM_LATENT_L, z_l)

    else:
        raise('NotImplementedError')

    return output

def ConcatOperator(z_l_0, z_l_1_pre, name):
    output = tf.concat([z_l_0, z_l_1_pre], axis=1)
    output = lib.ops.linear.Linear(name+'.Input', DIM_LATENT_L*2, DIM_OP, output)
    output = LeakyReLU(output)

    output = lib.ops.linear.Linear(name+'.1', DIM_OP, DIM_OP, output)
    output = LeakyReLU(output)

    output = lib.ops.linear.Linear(name+'.Output', DIM_OP, DIM_LATENT_L, output)

    if OP_DYN_MODE == 'res':
        output = z_l_0 + output
    
    elif OP_DYN_MODE == 'res_w':
        output = output + lib.ops.linear.Linear(name+'.ZW', DIM_LATENT_L, DIM_LATENT_L, z_l_0)

    else:
        raise('NotImplementedError')

    return output

def DynamicGenerator(z_l_0):
    z_list = [z_l_0,]

    epsilon = tf.random_normal([BATCH_SIZE, DIM_LATENT_T])
    for i in xrange(LEN-1):
        z_list.append(ImplicitOperator(z_list[-1], epsilon, 'Generator.Dynamic'))

    return tf.reshape(tf.concat(z_list, axis=1), [BATCH_SIZE, LEN, DIM_LATENT_L])

def DynamicExtractor(z_l_pre):
    if POS_MODE is 'inverse':
        z_list = [z_l_pre[:,LEN - 1,:],]
        for i in xrange(LEN-1):
            z_list.insert(0, ConcatOperator(z_list[0], z_l_pre[:,LEN - i - 2,:], 'Extractor.Dynamic.Backward'))

    elif POS_MODE is 'forward_inverse':
        z_list = [z_l_pre[:,0,:],]
        for i in xrange(LEN-1):
            z_list.append(ConcatOperator(z_list[-1], z_l_pre[:,i + 1,:], 'Extractor.Dynamic.Forward'))

    elif POS_MODE is 'gsp':
        tmp_z_list = [z_l_pre[:,LEN - 1,:],]
        for i in xrange(LEN-1):
            tmp_z_list.insert(0, ConcatOperator(tmp_z_list[0], z_l_pre[:,LEN - i - 2,:], 'Extractor.Dynamic.Backward'))
        z_list = [tmp_z_list[0],]
        for i in xrange(LEN-1):
            z_list.append(ConcatOperator(z_list[-1], tmp_z_list[i + 1], 'Extractor.Dynamic.Forward'))

    elif POS_MODE is 'naive_mean_field':
        return z_l_pre

    else:
        raise('NotImplementedError')

    return tf.reshape(tf.concat(z_list, axis=1), [BATCH_SIZE, LEN, DIM_LATENT_L])

def Generator(z_g, z_l):
    z_g = tf.reshape(z_g, [BATCH_SIZE, DIM_LATENT_G])
    z_g = tf.tile(tf.expand_dims(z_g, axis=1), [1, LEN, 1])
    z_l = tf.reshape(z_l, [BATCH_SIZE, LEN, DIM_LATENT_L])
    z = tf.concat([z_g, z_l], axis=-1)
    z = tf.reshape(z, [BATCH_SIZE*LEN, DIM_LATENT_G+DIM_LATENT_L])

    output = lib.ops.linear.Linear('Generator.Input', DIM_LATENT_G+DIM_LATENT_L, 4*4*8*DIM, z)
    if BN_FLAG_G:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [BATCH_SIZE*LEN, 8*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*DIM, 4*DIM, 5, output)
    if BN_FLAG_G:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*DIM, 2*DIM, 5, output)
    if BN_FLAG_G:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*DIM, DIM, 5, output)
    if BN_FLAG_G:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN4', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)
    output = tf.tanh(output)

    return tf.reshape(output, [BATCH_SIZE, LEN, OUTPUT_DIM])

def Extractor(inputs):
    output = tf.reshape(inputs, [BATCH_SIZE*LEN,] + OUTPUT_SHAPE)

    output = lib.ops.conv2d.Conv2D('Extractor.1', 3, DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Extractor.2', DIM, 2*DIM, 5, output, stride=2)
    if BN_FLAG_E:
        output = lib.ops.batchnorm.Batchnorm('Extractor.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Extractor.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if BN_FLAG_E:
        output = lib.ops.batchnorm.Batchnorm('Extractor.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Extractor.4', 4*DIM, 8*DIM, 5, output, stride=2)
    if BN_FLAG_E:
        output = lib.ops.batchnorm.Batchnorm('Extractor.BN4', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [BATCH_SIZE*LEN, 4*4*8*DIM])

    output = lib.ops.linear.Linear('Extractor.Output', 4*4*8*DIM, DIM_LATENT_L, output)

    return tf.reshape(output, [BATCH_SIZE, LEN, DIM_LATENT_L])

def G_Extractor(inputs):
    output = tf.reshape(inputs, [BATCH_SIZE, 3*LEN, 64, 64])

    output = lib.ops.conv2d.Conv2D('Extractor.G.1', 3*LEN, DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Extractor.G.2', DIM, 2*DIM, 5, output, stride=2)
    if BN_FLAG_E:
        output = lib.ops.batchnorm.Batchnorm('Extractor.G.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Extractor.G.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if BN_FLAG_E:
        output = lib.ops.batchnorm.Batchnorm('Extractor.G.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Extractor.G.4', 4*DIM, 8*DIM, 5, output, stride=2)
    if BN_FLAG_E:
        output = lib.ops.batchnorm.Batchnorm('Extractor.G.BN4', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [BATCH_SIZE, 4*4*8*DIM])
    output = lib.ops.linear.Linear('Extractor.G.Output', 4*4*8*DIM, DIM_LATENT_G, output)

    return tf.reshape(output, [BATCH_SIZE, DIM_LATENT_G])

if MODE in ['local_ep', 'local_epce-z']:
    def Discriminator(x, z_g, z_l):
        output = tf.reshape(x, [BATCH_SIZE*LEN,] + OUTPUT_SHAPE)
        z_g = tf.reshape(z_g, [BATCH_SIZE, DIM_LATENT_G])
        z_g = tf.tile(tf.expand_dims(z_g, axis=1), [1, LEN, 1])
        z_l = tf.reshape(z_l, [BATCH_SIZE, LEN, DIM_LATENT_L])
        z = tf.concat([z_g, z_l], axis=-1)
        z = tf.reshape(z, [BATCH_SIZE*LEN, DIM_LATENT_G+DIM_LATENT_L])

        output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, DIM, 5,output, stride=2)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
        if BN_FLAG_D:
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
        if BN_FLAG_D:
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*DIM, 8*DIM, 5, output, stride=2)
        if BN_FLAG_D:
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN4', [0,2,3], output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = tf.reshape(output, [BATCH_SIZE*LEN, 4*4*8*DIM])

        z_output = lib.ops.linear.Linear('Discriminator.z1', DIM_LATENT_G+DIM_LATENT_L, 512, z)
        z_output = LeakyReLU(z_output)
        z_output = tf.layers.dropout(z_output, rate=.2)

        output = tf.concat([output, z_output], 1)
        output = lib.ops.linear.Linear('Discriminator.zx1', 4*4*8*DIM+512, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.linear.Linear('Discriminator.Output', 512, 1, output)

        return tf.reshape(output, [BATCH_SIZE*LEN,])

    def DynamicDiscrminator(z1, z2):
        z1 = tf.reshape(z1, [BATCH_SIZE, DIM_LATENT_L])
        z2 = tf.reshape(z2, [BATCH_SIZE, DIM_LATENT_L])
        output = tf.concat([z1, z2], axis=1)
        output = lib.ops.linear.Linear('Discriminator.Dynamic.Input', DIM_LATENT_L*2, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.linear.Linear('Discriminator.Dynamic.2', 512, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.linear.Linear('Discriminator.Dynamic.3', 512, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.linear.Linear('Discriminator.Dynamic.Output', 512, 1, output)

        return tf.reshape(output, [BATCH_SIZE,])

    def ZGDiscrminator(z_g):
        output = tf.reshape(z_g, [BATCH_SIZE, DIM_LATENT_G])
        output = lib.ops.linear.Linear('Discriminator.ZG.Input', DIM_LATENT_G, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.linear.Linear('Discriminator.ZG.2', 512, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.linear.Linear('Discriminator.ZG.3', 512, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.linear.Linear('Discriminator.ZG.Output', 512, 1, output)

        return tf.reshape(output, [BATCH_SIZE,])

elif MODE in ['ali', 'alice-z']:
    if ALI_MODE is '3dcnn':
        import tflib.ops.conv3d
        def Discriminator(x, z_g, z_l):
            output = tf.reshape(x, [-1, LEN] + OUTPUT_SHAPE)
            output = tf.transpose(output, [0, 1, 3, 4, 2]) #NLHWC

            z_l = tf.reshape(z_l, [BATCH_SIZE, LEN*DIM_LATENT_L])
            z_g = tf.reshape(z_g, [BATCH_SIZE, DIM_LATENT_G])
            z = tf.concat([z_g, z_l], axis=-1)

            if LEN == 31:
                output = lib.ops.conv3d.Conv3D('Discriminator.1', 4, 3, DIM, 4, output, stride=2, stride_len=4)
            elif LEN == 4:
                output = lib.ops.conv3d.Conv3D('Discriminator.1', 4, 3, DIM, 4, output, stride=2, stride_len=2)
            else:
                raise('NotImplementedError')

            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            if LEN == 4:
                output = lib.ops.conv3d.Conv3D('Discriminator.2', 4, DIM, 2*DIM, 4, output, stride=2, stride_len=1)
            elif LEN == 31:
                output = lib.ops.conv3d.Conv3D('Discriminator.2', 4, DIM, 2*DIM, 4, output, stride=2, stride_len=2)
            else:
                raise('NotImplementedError')

            if BN_FLAG_D:
                output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,1,2,3], output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.conv3d.Conv3D('Discriminator.3', 4, 2*DIM, 4*DIM, 4, output, stride=2, stride_len=2)
            if BN_FLAG_D:
                output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,1,2,3], output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            if LEN == 4:
                output = lib.ops.conv3d.Conv3D('Discriminator.4', 4, 4*DIM, 8*DIM, 4, output, stride=2, stride_len=1)
            elif LEN == 31:
                output = lib.ops.conv3d.Conv3D('Discriminator.4', 4, 4*DIM, 8*DIM, 4, output, stride=2, stride_len=2)
            else:
                raise('NotImplementedError')

            if BN_FLAG_D:
                output = lib.ops.batchnorm.Batchnorm('Discriminator.BN4', [0,1,2,3], output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = tf.reshape(output, [BATCH_SIZE, 4*4*8*DIM])

            z_output = lib.ops.linear.Linear('Discriminator.z1', DIM_LATENT_G+DIM_LATENT_L*LEN, 512, z)
            z_output = LeakyReLU(z_output)
            z_output = tf.layers.dropout(z_output, rate=.2)

            output = tf.concat([output, z_output], 1)
            output = lib.ops.linear.Linear('Discriminator.zx1', 4*4*8*DIM+512, 512, output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.linear.Linear('Discriminator.Output', 512, 1, output)

            return tf.reshape(output, [BATCH_SIZE,])

    elif ALI_MODE is 'concat_x':
        def Discriminator(x, z_g, z_l):
            output = tf.reshape(x, [BATCH_SIZE, 3*LEN, 64, 64])
            
            z_l = tf.reshape(z_l, [BATCH_SIZE, LEN*DIM_LATENT_L])
            z_g = tf.reshape(z_g, [BATCH_SIZE, DIM_LATENT_G])
            z = tf.concat([z_g, z_l], axis=-1)

            output = lib.ops.conv2d.Conv2D('Discriminator.1', 3*LEN, DIM, 5, output, stride=2)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
            if BN_FLAG_D:
                output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
            if BN_FLAG_D:
                output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*DIM, 8*DIM, 5, output, stride=2)
            if BN_FLAG_D:
                output = lib.ops.batchnorm.Batchnorm('Discriminator.BN4', [0,2,3], output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = tf.reshape(output, [BATCH_SIZE, 4*4*8*DIM])

            z_output = lib.ops.linear.Linear('Discriminator.z1', DIM_LATENT_G+DIM_LATENT_L*LEN, 512, z)
            z_output = LeakyReLU(z_output)
            z_output = tf.layers.dropout(z_output, rate=.2)

            output = tf.concat([output, z_output], 1)
            output = lib.ops.linear.Linear('Discriminator.zx1', 4*4*8*DIM+512, 512, output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.linear.Linear('Discriminator.Output', 512, 1, output)

            return tf.reshape(output, [BATCH_SIZE,])

    elif ALI_MODE is 'concat_z':
        def Discriminator(x, z_g, z_l):
            output = tf.reshape(x, [BATCH_SIZE*LEN, 3, 64, 64])
            
            z_l = tf.reshape(z_l, [BATCH_SIZE, LEN*DIM_LATENT_L])
            z_g = tf.reshape(z_g, [BATCH_SIZE, DIM_LATENT_G])
            z = tf.concat([z_g, z_l], axis=-1)

            output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, DIM, 5, output, stride=2)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
            if BN_FLAG_D:
                output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
            if BN_FLAG_D:
                output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*DIM, 8*DIM, 5, output, stride=2)
            if BN_FLAG_D:
                output = lib.ops.batchnorm.Batchnorm('Discriminator.BN4', [0,2,3], output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.conv2d.Conv2D('Discriminator.5', 8*DIM, DIM_LATENT_G, 4, output, stride=1, padding='VALID')

            output = tf.reshape(output, [BATCH_SIZE, LEN*DIM_LATENT_G])

            z_output = lib.ops.linear.Linear('Discriminator.z1', DIM_LATENT_G+DIM_LATENT_L*LEN, 512, z)
            z_output = LeakyReLU(z_output)
            z_output = tf.layers.dropout(z_output, rate=.2)

            output = tf.concat([output, z_output], 1)
            output = lib.ops.linear.Linear('Discriminator.zx1', LEN*DIM_LATENT_G+512, 512, output)
            output = LeakyReLU(output)
            output = tf.layers.dropout(output, rate=.2)

            output = lib.ops.linear.Linear('Discriminator.Output', 512, 1, output)

            return tf.reshape(output, [BATCH_SIZE,])

else:
    raise('NotImplementedError')


'''
losses
'''
real_x_unit = tf.placeholder(tf.float32, shape=[BATCH_SIZE, LEN, OUTPUT_DIM])
real_x = 2*((tf.cast(real_x_unit, tf.float32)/256.)-.5)
q_z_l_pre = Extractor(real_x)
q_z_g = G_Extractor(real_x)
q_z_l = DynamicExtractor(q_z_l_pre)
rec_x = Generator(q_z_g, q_z_l)

p_z_l_0 = tf.random_normal([BATCH_SIZE, DIM_LATENT_L])
p_z_l = DynamicGenerator(p_z_l_0)
p_z_g = tf.random_normal([BATCH_SIZE, DIM_LATENT_G])
fake_x = Generator(p_z_g, p_z_l)

if MODE in ['local_ep', 'local_epce-z']:
    disc_fake, disc_real = [],[]
    for i in xrange(LEN-1):
        disc_fake.append(DynamicDiscrminator(p_z_l[:,i,:], p_z_l[:,i+1,:]))
        disc_real.append(DynamicDiscrminator(q_z_l[:,i,:], q_z_l[:,i+1,:]))
    disc_fake.append(ZGDiscrminator(p_z_g))
    disc_real.append(ZGDiscrminator(q_z_g))
    disc_fake.append(Discriminator(fake_x, p_z_g, p_z_l))
    disc_real.append(Discriminator(real_x, q_z_g, q_z_l))

elif MODE in ['ali', 'alice-z']:
    disc_real = Discriminator(real_x, q_z_g, q_z_l)
    disc_fake = Discriminator(fake_x, p_z_g, p_z_l)

gen_params = lib.params_with_name('Generator')
ext_params = lib.params_with_name('Extractor')
disc_params = lib.params_with_name('Discriminator')

if MODE == 'local_ep':
    rec_penalty = None
    gen_cost, disc_cost, _, _, gen_train_op, disc_train_op = lib.objs.gan_inference.weighted_ampce(disc_fake, disc_real, ratio, gen_params+ext_params, disc_params, lr=LR, beta1=BETA1, rec_penalty=rec_penalty)

elif MODE == 'local_epce-z':
    rec_penalty = LAMBDA*lib.utils.distance.distance(real_x, rec_x, 'l2')
    gen_cost, disc_cost, _, _, gen_train_op, disc_train_op = lib.objs.gan_inference.weighted_ampce(disc_fake, disc_real, ratio, gen_params+ext_params, disc_params, lr=LR, beta1=BETA1, rec_penalty=rec_penalty)

elif MODE == 'ali':
    rec_penalty = None
    gen_cost, disc_cost, gen_train_op, disc_train_op = lib.objs.gan_inference.ali(disc_fake, disc_real, gen_params+ext_params, disc_params, lr=LR, beta1=BETA1, beta2=BETA2)

elif MODE == 'alice-z':
    rec_penalty = LAMBDA*lib.utils.distance.distance(real_x, rec_x, 'l2')
    gen_cost, disc_cost, gen_train_op, disc_train_op = lib.objs.gan_inference.alice(disc_fake, disc_real, rec_penalty, gen_params+ext_params, disc_params, lr=LR, beta1=BETA1)

# Dataset iterator
train_gen, dev_gen = lib.chairs.load(seq_length=LEN, batch_size=BATCH_SIZE, size=64, data_dir=DATA_DIR, num_dev=200)
def inf_train_gen():
    while True:
        for images in train_gen():
            yield images

# For visualization
def vis(x, iteration, num, name):
    lib.save_images.save_images(
        x.reshape((-1,) + tuple(OUTPUT_SHAPE)), 
        os.path.join(outf, name+'_'+str(iteration)+'.png'), 
        size = (num, LEN)
    )
    x = x.reshape((num, LEN, 3, 64, 64))
    lib.save_images.save_gifs(x, os.path.join(outf, name+'_'+str(iteration)+'.gif'), size=None)

# For generation
pre_fixed_noise = tf.constant(np.random.normal(size=(N_VIS, DIM_LATENT_L)).astype('float32'))
fixed_noise_g = tf.constant(np.random.normal(size=(N_VIS, DIM_LATENT_G)).astype('float32'))
fixed_noise_l = DynamicGenerator(pre_fixed_noise)
fixed_noise_samples = Generator(fixed_noise_g, fixed_noise_l)
def generate_video(iteration, data):
    samples = session.run(fixed_noise_samples)
    samples = ((samples+1.)*(255.99/2)).astype('int32')
    vis(samples, iteration, N_VIS, 'samples')
    vis(data, iteration, BATCH_SIZE, 'train_data')

# For reconstruction
fixed_data = dev_gen().next()
def reconstruct_video(iteration):
    rec_samples = session.run(rec_x, feed_dict={real_x_unit: fixed_data})
    rec_samples = ((rec_samples+1.)*(255.99/2)).astype('int32')
    rec_samples = rec_samples.reshape((-1, LEN, OUTPUT_DIM))
    tmp_list = []
    for i in xrange(BATCH_SIZE):
        tmp_list.append(fixed_data[i])
        tmp_list.append(rec_samples[i])
    rec_samples = np.vstack(tmp_list)
    vis(rec_samples, iteration, BATCH_SIZE*2, 'reconstruction')

# disentangle
fixed_data = dev_gen().next()
dis_g = tf.constant(np.tile(np.random.normal(size=(1, DIM_LATENT_G)).astype('float32'), [BATCH_SIZE, 1]))
dis_x = Generator(dis_g, q_z_l)
def disentangle(iteration):
    samples = session.run(dis_x, feed_dict={real_x_unit: fixed_data})
    samples = ((samples+1.)*(255.99/2)).astype('int32')
    tmp_list = []
    for i in xrange(BATCH_SIZE):
        tmp_list.append(fixed_data[i])
        tmp_list.append(samples[i])
    samples = np.vstack(tmp_list)
    vis(samples, iteration, BATCH_SIZE*2, 'disentangle')

'''
Train loop
'''
saver = tf.train.Saver()
with tf.Session() as session:

    session.run(tf.global_variables_initializer())
    gen = inf_train_gen()

    total_num = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    print '\nTotol number of parameters', total_num
    with open(logfile,'a') as f:
        f.write('\nTotol number of parameters' + str(total_num) + '\n')

    gen_num = tf.reduce_sum([tf.reduce_prod(tf.shape(t)) for t in gen_params])
    ext_num = tf.reduce_sum([tf.reduce_prod(tf.shape(t)) for t in ext_params])
    disc_num = tf.reduce_sum([tf.reduce_prod(tf.shape(t)) for t in disc_params])

    print '\nNumber of parameters in each player', session.run([gen_num, ext_num, disc_num, gen_num+ext_num+disc_num]), '\n'
    with open(logfile,'a') as f:
        f.write('\nNumber of parameters in each player' + str(session.run([gen_num, ext_num, disc_num, gen_num+ext_num+disc_num])) + '\n')

    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:
            _data = gen.next()
            if rec_penalty is None:
                _gen_cost, _ = session.run([gen_cost, gen_train_op],
                feed_dict={real_x_unit: _data})
            else:
                _gen_cost, _rec_cost, _ = session.run([gen_cost, rec_penalty, gen_train_op],
                feed_dict={real_x_unit: _data})
            
        for i in xrange(CRITIC_ITERS):
            _data = gen.next()
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_x_unit: _data}
            )
        if iteration > 0:
            lib.plot.plot('gc', _gen_cost)
            if rec_penalty is not None:
                lib.plot.plot('rc', _rec_cost)
        lib.plot.plot('dc', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        # Write logs
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush(outf, logfile)
        lib.plot.tick()

        # Generation and reconstruction
        if iteration % 5000 == 4999:
            generate_video(iteration, _data)
            reconstruct_video(iteration)
            disentangle(iteration)

        # Save model
        if iteration == ITERS - 1:
            save_path = saver.save(session, os.path.join(outf, '{}_model_{}.ckpt'.format(iteration, MODE)))