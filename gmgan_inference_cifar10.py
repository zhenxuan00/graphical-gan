import os, sys, shutil, time
sys.path.append(os.getcwd())

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
import tflib.cifar10
import tflib.inception_score
import tflib.plot
import tflib.visualization
import tflib.objs.gan_inference
import tflib.objs.mmd
import tflib.objs.kl
import tflib.objs.kl_aggregated
import tflib.objs.discrete_variables
import tflib.utils.distance

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = './dataset/cifar10/cifar-10-batches-py'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

'''
hyperparameters
'''
MODE = 'local_ep' # ali, local_ep, alice, local_epce, vegan
if MODE in ['vegan-kl', 'vegan-ikl', 'vegan-jsd']:
    TYPE_Q = 'learn_std' # learn_std, fix_std, no_std
    TYPE_P = 'no_std'
    Z_SAMPLES = 100 # MC estimation for D(q(z)||p(z))
elif MODE is 'vae':
    TYPE_Q = 'learn_std'
    TYPE_P = 'learn_std'
else:
    TYPE_Q = 'no_std'
    TYPE_P = 'no_std'
STD = .1 # For fix_std
d_list = ['alice', 'alice-z', 'alice-x', 'vegan', 'vegan-wgan-gp', 'vegan-kl', 'vegan-ikl', 'vegan-jsd', 'vegan-mmd', 'local_epce']
if MODE in d_list:
    DISTANCE_X = 'l2' # l1, l2
if MODE in ['vegan-mmd', 'vegan-kl', 'vegan-ikl', 'vegan-jsd', 'vae']:
    CRITIC_ITERS = 0 # No discriminators
elif MODE in ['vegan', 'vegan-wgan-gp', 'wali', 'wali-gp']:
    CRITIC_ITERS = 5 # 5 iters of D per iter of G
else:
    CRITIC_ITERS = 1

BATCH_SIZE = 64 # Batch size
LAMBDA = 1. # Balance reconstruction and regularization in vegan
LR = 2e-4
if MODE in ['vae']:
    BETA1 = .9
else:
    BETA1 = .5
ITERS = 200000 # How many generator iterations to train for

DIM = 64 # Model dimensionality
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)
if MODE in ['vegan', 'vegan-wgan-gp', 'vegan-kl', 'vegan-jsd', 'vegan-ikl']:
    BN_FLAG = False # Use batch_norm or not
    DIM_LATENT = 8 # Dimensionality of the latent z
else:
    BN_FLAG = True 
    DIM_LATENT = 128
N_COMS = 30
N_VIS = N_COMS*10 # Number of samples to be visualized
assert(N_VIS%N_COMS==0)
MODE_K = 'CONCRETE' # CONCRETE, REINFORCE, STRAIGHT_THROUGHT_CONCRETE, STRAIGHT_THROUGHT
if MODE_K is 'REINFORCE':
    CONTROL_VARIATE = .0
elif MODE_K in ['CONCRETE', 'STRAIGHT_THROUGHT_CONCRETE']:
    TEMP_INIT = .1
    TEMP = TEMP_INIT
DR_RATE = .2


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
unit_std_x = tf.constant((STD*np.ones(shape=(BATCH_SIZE, OUTPUT_DIM))).astype('float32'))
unit_std_z = tf.constant((STD*np.ones(shape=(BATCH_SIZE, DIM_LATENT))).astype('float32'))
### prior
PI = tf.constant(np.asarray([1./N_COMS,]*N_COMS, dtype=np.float32))
prior_k = tf.distributions.Categorical(probs=PI)

def sample_gumbel(shape, eps=1e-20): 
  # Sample from Gumbel(0, 1)
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def GaussianNoiseLayer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise

### Very simple MoG
def HyperGenerator(hyper_k, hyper_noise):
    com_mu = lib.param('Generator.Hyper.Mu', np.random.normal(size=(N_COMS, DIM_LATENT)).astype('float32'))
    noise = tf.add(tf.matmul(tf.cast(hyper_k, tf.float32), com_mu), hyper_noise)
    return noise

### Very simple soft alignment
def HyperExtractor(latent_z):
    com_mu = lib.param('Generator.Hyper.Mu', np.random.normal(size=(N_COMS, DIM_LATENT)).astype('float32'))
    com_logits = -.5*tf.reduce_sum(tf.pow((tf.expand_dims(latent_z, axis=1) - tf.expand_dims(com_mu, axis=0)), 2), axis=-1) + tf.expand_dims(tf.log(PI), axis=0)

    if MODE_K is 'REINFORCE':
        k = tf.one_hot(indices=tf.argmax(com_logits, axis=-1), depth=N_COMS)
    elif MODE_K is 'CONCRETE':
        k = tf.nn.softmax((com_logits + sample_gumbel(tf.shape(com_logits)))/TEMP)
    elif MODE_K is 'STRAIGHT_THROUGHT_CONCRETE':
        k = tf.nn.softmax((com_logits + sample_gumbel(tf.shape(com_logits)))/TEMP)
        k_hard = tf.one_hot(indices=tf.argmax(k, axis=-1), depth=N_COMS)
        k = tf.stop_gradient(k_hard - k) + k

    elif MODE_K is 'STRAIGHT_THROUGHT':
        k_hard = tf.one_hot(indices=tf.argmax(com_logits, axis=-1), depth=N_COMS)
        k = tf.stop_gradient(k_hard - com_logits) + com_logits

    return com_logits, k

def Generator(noise):
    output = lib.ops.linear.Linear('Generator.Input', DIM_LATENT, 4*4*4*DIM, noise)
    if BN_FLAG:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    if BN_FLAG:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    if BN_FLAG:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM]), None, None

def Extractor(inputs):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = lib.ops.conv2d.Conv2D('Extractor.1', 3, DIM, 5, output,stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Extractor.2', DIM, 2*DIM, 5, output, stride=2)
    if BN_FLAG:
        output = lib.ops.batchnorm.Batchnorm('Extractor.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Extractor.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if BN_FLAG:
        output = lib.ops.batchnorm.Batchnorm('Extractor.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])

    if TYPE_Q is 'learn_std':
        log_std = lib.ops.linear.Linear('Extractor.Std', 4*4*4*DIM, DIM_LATENT, output)
        std = tf.exp(log_std)
    elif TYPE_Q is 'fix_std':
        std = unit_std_z
    else:
        std = None
        mean = None

    output = lib.ops.linear.Linear('Extractor.Output', 4*4*4*DIM, DIM_LATENT, output)
    
    if TYPE_Q in ['learn_std', 'fix_std']:
        epsilon = tf.random_normal(unit_std_z.shape)
        mean = output
        output = tf.add(mean, tf.multiply(epsilon, std))

    return tf.reshape(output, [-1, DIM_LATENT]), mean, std

if MODE in ['vegan', 'vegan-wgan-gp']:

    def Discriminator(z, k):
        output = tf.concat([z, k], 1)
        output = lib.ops.linear.Linear('Discriminator.HyperInput', DIM_LATENT+N_COMS, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=DR_RATE)

        output = lib.ops.linear.Linear('Discriminator.Hyper2', 512, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=DR_RATE)

        output = lib.ops.linear.Linear('Discriminator.Hyper3', 512, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=DR_RATE)

        output = lib.ops.linear.Linear('Discriminator.HyperOutput', 512, 1, output)

        return tf.reshape(output, [-1])

elif MODE in ['local_ep', 'local_epce']:

    def HyperDiscriminator(z, k):
        output = tf.concat([z, k], 1)
        output = lib.ops.linear.Linear('Discriminator.HyperInput', DIM_LATENT+N_COMS, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=DR_RATE)

        output = lib.ops.linear.Linear('Discriminator.Hyper2', 512, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=DR_RATE)

        output = lib.ops.linear.Linear('Discriminator.Hyper3', 512, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=DR_RATE)

        output = lib.ops.linear.Linear('Discriminator.HyperOutput', 512, 1, output)

        return tf.reshape(output, [-1])

    def Discriminator(x, z):
        output = tf.reshape(x, [-1, 3, 32, 32])

        output = lib.ops.conv2d.Conv2D('Discriminator.1',3,DIM,5,output,stride=2)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=DR_RATE)

        output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=DR_RATE)

        output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=DR_RATE)

        output = tf.reshape(output, [-1, 4*4*4*DIM])

        z_output = lib.ops.linear.Linear('Discriminator.z1', DIM_LATENT, 512, z)
        z_output = LeakyReLU(z_output)
        z_output = tf.layers.dropout(z_output, rate=DR_RATE)

        output = tf.concat([output, z_output], 1)
        output = lib.ops.linear.Linear('Discriminator.zx1', 4*4*4*DIM+512, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=DR_RATE)

        output = lib.ops.linear.Linear('Discriminator.Output', 512, 1, output)

        return tf.reshape(output, [-1])

elif MODE in ['vegan-mmd', 'vegan-kl', 'vegan-ikl', 'vegan-jsd', 'vae']:
    pass # no discriminator

else:
    def Discriminator(x, z, k):
        output = tf.reshape(x, [-1, 3, 32, 32])

        output = lib.ops.conv2d.Conv2D('Discriminator.x1',3,DIM,5,output,stride=2)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=DR_RATE)

        output = lib.ops.conv2d.Conv2D('Discriminator.x2', DIM, 2*DIM, 5, output, stride=2)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=DR_RATE)

        output = lib.ops.conv2d.Conv2D('Discriminator.x3', 2*DIM, 4*DIM, 5, output, stride=2)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=DR_RATE)

        output = tf.reshape(output, [-1, 4*4*4*DIM])

        zk_output = tf.concat([z, k], 1)
        zk_output = lib.ops.linear.Linear('Discriminator.zk1', DIM_LATENT+N_COMS, 512, zk_output)
        zk_output = LeakyReLU(zk_output)
        zk_output = tf.layers.dropout(zk_output, rate=DR_RATE)

        output = tf.concat([output, zk_output], 1)
        output = lib.ops.linear.Linear('Discriminator.zkx1', 4*4*4*DIM+512, 512, output)
        output = LeakyReLU(output)
        output = tf.layers.dropout(output, rate=DR_RATE)

        output = lib.ops.linear.Linear('Discriminator.Output', 512, 1, output)

        return tf.reshape(output, [-1])

'''
losses
'''
real_x_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
real_x = 2*((tf.cast(real_x_int, tf.float32)/255.)-.5)
q_z, q_z_mean, q_z_std = Extractor(real_x)
q_k_logits, q_k = HyperExtractor(q_z)
q_k_probs = tf.nn.softmax(q_k_logits)
if MODE_K is 'REINFORCE':
    q_k_prob_max = tf.reduce_max(q_k_probs, axis=1)
rec_x, rec_x_mean, rec_x_std = Generator(q_z)
hyper_p_z = tf.random_normal([BATCH_SIZE, DIM_LATENT])
hyper_p_k = tf.one_hot(indices=prior_k.sample(BATCH_SIZE), depth=N_COMS)
p_z = HyperGenerator(hyper_p_k, hyper_p_z)
fake_x, _, _ = Generator(p_z)
rec_z, _, _ = Extractor(fake_x)
rec_q_k_logits, rec_q_k = HyperExtractor(rec_z)

if MODE_K is not 'REINFORCE':
    score_function = None

if MODE is 'vegan':
    disc_fake = Discriminator(p_z, hyper_p_k)
    disc_real = Discriminator(q_z, q_k)
    if MODE_K is 'REINFORCE':
        score_function = lib.objs.discrete_variables.score_function(disc_real, q_k_prob_max, CONTROL_VARIATE)

elif MODE in ['local_ep', 'local_epce']:
    disc_fake, disc_real = [],[]
    disc_fake.append(HyperDiscriminator(p_z, hyper_p_k))
    disc_real.append(HyperDiscriminator(q_z, q_k))
    disc_fake.append(Discriminator(fake_x, p_z))
    disc_real.append(Discriminator(real_x, q_z))

    if MODE_K is 'REINFORCE':
        score_function = lib.objs.discrete_variables.score_function(disc_real[0], q_k_prob_max, CONTROL_VARIATE)

else:
    disc_real = Discriminator(real_x, q_z, q_k)
    disc_fake = Discriminator(fake_x, p_z, hyper_p_k)
    if MODE_K is 'REINFORCE':
        score_function = lib.objs.discrete_variables.score_function(disc_real, q_k_prob_max, CONTROL_VARIATE)

gen_params = lib.params_with_name('Generator')
ext_params = lib.params_with_name('Extractor')
disc_params = lib.params_with_name('Discriminator')

if MODE == 'ali':
    rec_penalty = None
    gen_cost, disc_cost, gen_train_op, disc_train_op = lib.objs.gan_inference.ali(disc_fake, disc_real, gen_params+ext_params, disc_params, lr=LR, beta1=BETA1, s_f=score_function)

elif MODE == 'alice':
    rec_penalty = 1.*lib.utils.distance.distance(real_x, rec_x, DISTANCE_X)
    # rec_penalty += 1.*lib.utils.distance.distance(p_z, rec_z, DISTANCE_X)
    # rec_penalty += 1.*tf.nn.softmax_cross_entropy_with_logits(labels=hyper_p_k, logits=rec_q_k_logits)
    gen_cost, disc_cost, gen_train_op, disc_train_op = lib.objs.gan_inference.alice(disc_fake, disc_real, rec_penalty, gen_params+ext_params, disc_params, lr=LR, beta1=BETA1, s_f=score_function)

elif MODE == 'local_ep':
    rec_penalty = None
    gen_cost, disc_cost, gen_train_op, disc_train_op = lib.objs.gan_inference.local_ep(disc_fake, disc_real, gen_params+ext_params, disc_params, lr=LR, beta1=BETA1, s_f=score_function)

elif MODE  == 'local_epce':
    rec_penalty = 1.*lib.utils.distance.distance(real_x, rec_x, DISTANCE_X)
    # rec_penalty += 1.*lib.utils.distance.distance(p_z, rec_z, DISTANCE_X)
    # rec_penalty += 1.*tf.nn.softmax_cross_entropy_with_logits(labels=hyper_p_k, logits=rec_q_k_logits)
    gen_cost, disc_cost, gen_train_op, disc_train_op = lib.objs.gan_inference.local_epce(disc_fake, disc_real, rec_penalty, gen_params+ext_params, disc_params, lr=LR, beta1=BETA1, s_f=score_function)

elif MODE == 'vegan':
    rec_penalty = 1.*lib.utils.distance.distance(real_x, rec_x, DISTANCE_X)
    gen_cost, disc_cost, gen_train_op, disc_train_op = lib.objs.gan_inference.vegan(disc_fake, disc_real, rec_penalty, gen_params+ext_params, disc_params, LAMBDA,lr=LR, beta1=BETA1, s_f=score_function)

else:
    raise('NotImplementedError')

# For visualizing samples
# np_fixed_noise = np.repeat(np.random.normal(size=(N_VIS/N_COMS, DIM_LATENT)).astype('float32'), N_COMS, axis=0)
np_fixed_noise = np.random.normal(size=(N_VIS, DIM_LATENT)).astype('float32')
np_fixed_k = np.tile(np.eye(N_COMS, dtype=int), (N_VIS/N_COMS, 1))
hyper_fixed_noise = tf.constant(np_fixed_noise)
hyper_fixed_k = tf.constant(np_fixed_k)
fixed_noise = HyperGenerator(hyper_fixed_k, hyper_fixed_noise)
fixed_noise_samples, _, _ = Generator(fixed_noise)
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    samples = ((samples+1.)*(255./2)).astype('int32')
    lib.save_images.save_images(
        samples.reshape((-1, 3, 32, 32)), 
        os.path.join(outf, '{}_samples_{}.png'.format(frame, MODE)),
        size = [N_VIS/N_COMS, N_COMS]
    )

# For calculating inception score
hyper_p_z_is = tf.random_normal([BATCH_SIZE, DIM_LATENT])
hyper_p_k_is = tf.one_hot(indices=prior_k.sample(BATCH_SIZE), depth=N_COMS)
p_z_is = HyperGenerator(hyper_p_k_is, hyper_p_z_is)
samples_is, _, _ = Generator(p_z_is)
def get_inception_score():
    all_samples = []
    for i in xrange(50000/BATCH_SIZE):
        all_samples.append(session.run(samples_is))
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = ((all_samples+1.)*(255./2)).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
    return lib.inception_score.get_inception_score(list(all_samples))

# Dataset iterator
train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, data_dir=DATA_DIR)
def inf_train_gen():
    while True:
        for images, targets in train_gen():
            yield images

# For reconstruction
fixed_data_int = lib.cifar10.get_reconstruction_data(BATCH_SIZE, data_dir=DATA_DIR)
rand_data_int, _ = dev_gen().next()
def reconstruct_image(frame, data_int):
    rec_samples = session.run(rec_x, feed_dict={real_x_int: data_int})
    rec_samples = ((rec_samples+1.)*(255./2)).astype('int32')
    tmp_list = []
    for d, r in zip(data_int, rec_samples):
        tmp_list.append(d)
        tmp_list.append(r)
    rec_samples = np.vstack(tmp_list)
    lib.save_images.save_images(
        rec_samples.reshape((-1, 3, 32, 32)), 
        os.path.join(outf, '{}_reconstruction_{}.png'.format(MODE, frame))
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
            if MODE is 'wali':
                _ = session.run(clip_disc_weights)
        
        if MODE in ['vegan-mmd', 'vegan-kl', 'vegan-ikl', 'vegan-jsd', 'vae']:
            if iteration > 0:
                lib.plot.plot('train gen cost ', _gen_cost)
        else:
            lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        # Calculate dev loss
        if iteration % 100 == 99:
            if rec_penalty is not None:
                dev_rec_costs = []
                dev_reg_costs = []
                for images,_ in dev_gen():
                    _dev_rec_cost, _dev_gen_cost = session.run(
                        [rec_penalty, gen_cost], 
                        feed_dict={real_x_int: images}
                    )
                    dev_rec_costs.append(_dev_rec_cost)
                    dev_reg_costs.append(_dev_gen_cost - _dev_rec_cost)
                lib.plot.plot('dev rec cost', np.mean(dev_rec_costs))
                lib.plot.plot('dev reg cost', np.mean(dev_reg_costs))
            else:
                dev_gen_costs = []
                for images,_ in dev_gen():
                    _dev_gen_cost = session.run(
                        gen_cost, 
                        feed_dict={real_x_int: images}
                    )
                    dev_gen_costs.append(_dev_gen_cost)
                lib.plot.plot('dev gen cost', np.mean(dev_gen_costs))

        # Write logs
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush(outf, logfile)
        
        # Calculate inception score
        if iteration % 10000 == 9999:
            inception_score = get_inception_score()
            lib.plot.plot('inception score', inception_score[0])
            lib.plot.plot('inception score std', inception_score[1])

        lib.plot.tick()

        # Generation and reconstruction
        if iteration % 5000 == 4999:
            generate_image(iteration, _data)
            reconstruct_image(iteration, rand_data_int)
            reconstruct_image(-iteration, fixed_data_int)

        # Save model
        if iteration == ITERS - 1:
            save_path = saver.save(session, os.path.join(outf, '{}_model_{}.ckpt'.format(iteration, MODE)))