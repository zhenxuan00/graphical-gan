import os, sys, shutil, time
sys.path.append(os.getcwd())

import functools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
from sklearn.manifold import TSNE
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.celebA
import tflib.plot
import tflib.visualization
import tflib.objs.gan_inference
import tflib.objs.mmd
import tflib.objs.kl
import tflib.objs.kl_aggregated
import tflib.objs.discrete_variables
import tflib.utils.distance

DATA_DIR = './dataset/celebA'


'''
hyperparameters
'''
MODE = 'local_ep' # ali, local_ep
CRITIC_ITERS = 1
BATCH_SIZE = 128 # Batch size
LR = 2e-4
DECAY = False
decay = 1.
BETA1 = .5
BETA2 = .999
ITERS = 100000 # How many generator iterations to train for

DIM_G = 32 # Model dimensionality
DIM_D = 32 # Model dimensionality
OUTPUT_DIM = 12288 # Number of pixels in celebA (3*64*64)
DIM_LATENT = 128
N_COMS = 100
N_VIS = N_COMS*10 # Number of samples to be visualized
assert(N_VIS%N_COMS==0)
MODE_K = 'CONCRETE'
TEMP_INIT = .1
TEMP = TEMP_INIT


'''
logs
'''
filename_script=os.path.basename(os.path.realpath(__file__))
outf=os.path.join("result", os.path.splitext(filename_script)[0])
outf+='.MODE-'
outf+=MODE
outf+='.N_COMS-'
outf+=str(N_COMS)
outf+='.'
outf+=str(int(time.time()))
if not os.path.exists(outf):
    os.makedirs(outf)
logfile=os.path.join(outf, 'logfile.txt')
shutil.copy(os.path.realpath(__file__), os.path.join(outf, filename_script))
lib.print_model_settings_to_file(locals().copy(), logfile)


'''
models
'''
### prior
PI = tf.constant(np.asarray([1./N_COMS,]*N_COMS, dtype=np.float32))
prior_k = tf.distributions.Categorical(probs=PI)

def sample_gumbel(shape, eps=1e-20): 
  # Sample from Gumbel(0, 1)
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def nonlinearity(x):
    return tf.nn.relu(x)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

### Very simple MoG
def HyperGenerator(hyper_k, hyper_noise):
    com_mu = lib.param('Generator.Hyper.Mu', np.random.normal(size=(N_COMS, DIM_LATENT)).astype('float32'))
    noise = tf.add(tf.matmul(tf.cast(hyper_k, tf.float32), com_mu), hyper_noise)
    return noise

### Very simple soft alignment
def HyperExtractor(latent_z):
    com_mu = lib.param('Generator.Hyper.Mu', np.random.normal(size=(N_COMS, DIM_LATENT)).astype('float32'))
    com_logits = -.5*tf.reduce_sum(tf.pow((tf.expand_dims(latent_z, axis=1) - tf.expand_dims(com_mu, axis=0)), 2), axis=-1) + tf.expand_dims(tf.log(PI), axis=0)

    k = tf.nn.softmax((com_logits + sample_gumbel(tf.shape(com_logits)))/TEMP)
    
    return com_logits, k

def Generator(noise):
    output = lib.ops.linear.Linear('Generator.Input', DIM_LATENT, 4*4*8*DIM_G, noise)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 8*DIM_G, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*DIM_G, 4*DIM_G, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*DIM_G, 2*DIM_G, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*DIM_G, DIM_G, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM_G, 3, 5, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Extractor(inputs):
    output = tf.reshape(inputs, [-1, 3, 64, 64])

    output = lib.ops.conv2d.Conv2D('Extractor.1', 3, DIM_G, 5, output,stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Extractor.2', DIM_G, 2*DIM_G, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Extractor.3', 2*DIM_G, 4*DIM_G, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Extractor.4', 4*DIM_G, 8*DIM_G, 5, output, stride=2)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*8*DIM_G])

    output = lib.ops.linear.Linear('Extractor.Output', 4*4*8*DIM_G, DIM_LATENT, output)
    
    return tf.reshape(output, [-1, DIM_LATENT])

if MODE in ['local_ep', 'local_epce']:

    def HyperDiscriminator(z, k):
        output = tf.concat([z, k], 1)
        output = lib.ops.linear.Linear('Discriminator.HyperInput', DIM_LATENT+N_COMS, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.linear.Linear('Discriminator.Hyper2', 512, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.linear.Linear('Discriminator.Hyper3', 512, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.linear.Linear('Discriminator.HyperOutput', 512, 1, output)

        return tf.reshape(output, [-1])

    def Discriminator(x, z):
        output = tf.reshape(x, [-1, 3, 64, 64])

        output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, DIM_D, 5,output, stride=2)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM_D, 2*DIM_D, 5, output, stride=2)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM_D, 4*DIM_D, 5, output, stride=2)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*DIM_D, 8*DIM_D, 5, output, stride=2)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = tf.reshape(output, [-1, 4*4*8*DIM_D])

        z_output = lib.ops.linear.Linear('Discriminator.z1', DIM_LATENT, 512, z)
        z_output = LeakyReLU(z_output)
        z_output = tf.layers.dropout(z_output, rate=.2)

        output = tf.concat([output, z_output], 1)
        output = lib.ops.linear.Linear('Discriminator.zx1', 4*4*8*DIM_D+512, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.linear.Linear('Discriminator.Output', 512, 1, output)

        return tf.reshape(output, [-1])

else:

    def Discriminator(x, z, k):
        output = tf.reshape(x, [-1, 3, 64, 64])

        output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, DIM_D, 5,output, stride=2)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM_D, 2*DIM_D, 5, output, stride=2)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM_D, 4*DIM_D, 5, output, stride=2)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*DIM_D, 8*DIM_D, 5, output, stride=2)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = tf.reshape(output, [-1, 4*4*8*DIM_D])

        zk_output = tf.concat([z, k], 1)
        zk_output = lib.ops.linear.Linear('Discriminator.zk1', DIM_LATENT+N_COMS, 512, zk_output)
        zk_output = LeakyReLU(zk_output)
        zk_output = tf.layers.dropout(zk_output, rate=.2)

        output = tf.concat([output, zk_output], 1)
        output = lib.ops.linear.Linear('Discriminator.zxk1', 4*4*8*DIM_D+512, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=.2)

        output = lib.ops.linear.Linear('Discriminator.Output', 512, 1, output)

        return tf.reshape(output, [-1])
'''
losses
'''
real_x_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
real_x = tf.reshape(2*((tf.cast(real_x_int, tf.float32)/256.)-.5), [BATCH_SIZE, OUTPUT_DIM])
real_x += tf.random_uniform(shape=[BATCH_SIZE,OUTPUT_DIM],minval=0.,maxval=1./128) # dequantize
q_z = Extractor(real_x)
q_k_logits, q_k = HyperExtractor(q_z)
q_k_probs = tf.nn.softmax(q_k_logits)
rec_x = Generator(q_z)
hyper_p_z = tf.random_normal([BATCH_SIZE, DIM_LATENT])
hyper_p_k = tf.one_hot(indices=prior_k.sample(BATCH_SIZE), depth=N_COMS)
p_z = HyperGenerator(hyper_p_k, hyper_p_z)
fake_x = Generator(p_z)

if MODE in ['local_ep', 'local_epce']:
    disc_fake, disc_real = [],[]
    disc_fake.append(HyperDiscriminator(p_z, hyper_p_k))
    disc_real.append(HyperDiscriminator(q_z, q_k))
    disc_fake.append(Discriminator(fake_x, p_z))
    disc_real.append(Discriminator(real_x, q_z))

else:
    disc_real = Discriminator(real_x, q_z, q_k)
    disc_fake = Discriminator(fake_x, p_z, hyper_p_k)

gen_params = lib.params_with_name('Generator')
ext_params = lib.params_with_name('Extractor')
disc_params = lib.params_with_name('Discriminator')

if MODE == 'ali':
    rec_penalty = None
    gen_cost, disc_cost, gen_train_op, disc_train_op = lib.objs.gan_inference.ali(disc_fake, disc_real, gen_params+ext_params, disc_params, lr=LR*decay, beta1=BETA1, beta2=BETA2)

elif MODE == 'local_ep':
    rec_penalty = None
    gen_cost, disc_cost, gen_train_op, disc_train_op = lib.objs.gan_inference.local_ep(disc_fake, disc_real, gen_params+ext_params, disc_params, lr=LR, beta1=BETA1, beta2=BETA2)

else:
    raise('NotImplementedError')

# For visualizing samples
# np_fixed_noise = np.repeat(np.random.normal(size=(N_VIS/N_COMS, DIM_LATENT)).astype('float32'), N_COMS, axis=0)
np_fixed_noise = np.random.normal(size=(N_VIS, DIM_LATENT)).astype('float32')
np_fixed_k = np.tile(np.eye(N_COMS, dtype=int), (N_VIS/N_COMS, 1))
hyper_fixed_noise = tf.constant(np_fixed_noise)
hyper_fixed_k = tf.constant(np_fixed_k)
fixed_noise = HyperGenerator(hyper_fixed_k, hyper_fixed_noise)
fixed_noise_samples = Generator(fixed_noise)
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    samples = ((samples+1.)*(255.99/2)).astype('int32')
    lib.save_images.save_images(
        samples.reshape((-1, 3, 64, 64)), 
        os.path.join(outf, '{}_samples_{}.png'.format(frame, MODE)),
        size = [N_VIS/N_COMS, N_COMS]
    )

# Dataset iterator
train_gen, dev_gen = lib.celebA.load(BATCH_SIZE, data_dir=DATA_DIR)
def inf_train_gen():
    while True:
        for images in train_gen():
            yield images

# For reconstruction
fixed_data_int = dev_gen().next()
def reconstruct_image(frame):    
    rec_samples = session.run(rec_x, feed_dict={real_x_int: fixed_data_int})
    rec_samples = ((rec_samples+1.)*(255.99/2)).astype('int32')
    tmp_list = []
    for d, r in zip(fixed_data_int, rec_samples):
        tmp_list.append(d)
        tmp_list.append(r)
    rec_samples = np.vstack(tmp_list)
    lib.save_images.save_images(
        rec_samples.reshape((-1, 3, 64, 64)), 
        os.path.join(outf, '{}_reconstruction_{}.png'.format(frame, MODE))
    )
saver = tf.train.Saver()


'''
Train loop
'''
with tf.Session() as session:

    session.run(tf.global_variables_initializer())
    gen = inf_train_gen()
    
    total_num = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    print '\nTotol number of parameters', total_num
    with open(logfile,'a') as f:
        f.write('Totol number of parameters' + str(total_num) + '\n')

    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:
            _data = gen.next()
            _gen_cost, _ = session.run([gen_cost, gen_train_op],
                feed_dict={real_x_int: _data}
            )
            
        for i in xrange(CRITIC_ITERS):
            _data = gen.next()
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_x_int: _data}
            )
        
        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        # Calculate dev loss
        if iteration % 100 == 99:
            dev_gen_costs = []
            for images in dev_gen():
                _dev_gen_cost = session.run(
                    gen_cost, 
                    feed_dict={real_x_int: images}
                )
                dev_gen_costs.append(_dev_gen_cost)
            lib.plot.plot('dev gen cost', np.mean(dev_gen_costs))
        
        # Write logs
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush(outf, logfile)
        lib.plot.tick()

        # Generation and reconstruction
        if iteration % 1000 == 999:
            generate_image(iteration, _data)
            reconstruct_image(iteration)
        
        # Save model
        if iteration == ITERS - 1:
            save_path = saver.save(session, os.path.join(outf, '{}_model_{}.ckpt'.format(iteration, MODE)))

        if DECAY:
            decay = tf.maximum(0., 1.-(tf.cast(iteration, tf.float32)/ITERS))