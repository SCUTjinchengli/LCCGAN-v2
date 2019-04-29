import argparse

parser = argparse.ArgumentParser()
# training parameters
parser.add_argument('--dataset', default='mnist', help='cifar10 | lsun | imagenet | folder | lfw | flowers')
parser.add_argument('--dataroot', default='./mnist',  help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize_s1', type=int, default=64, help='input batch size')
parser.add_argument('--batchSize_s2', type=int, default=1000, help='input batch size')
parser.add_argument('--batchSize_s3', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=5, help='size of the latent z vector')
parser.add_argument('--nc', type=int, default=3, help='the number of image channel')
parser.add_argument('--latent_dim', type=int, default=100, help='the number of image channel')
parser.add_argument('--anchor_num', type=int, default=128, help='the number of image channel')
parser.add_argument('--batchSize_inference', type=int, default=1, help='inference batch size')

# optimizer configuration
parser.add_argument('--s1_lr', type=float, default=0.0001, help='0.01 for lcc | 0.00001 for AE | 0.0002 for GAN')
parser.add_argument('--s2_lr', type=float, default=0.01, help='0.01 for lcc | 0.00001 for AE | 0.0002 for GAN')
parser.add_argument('--s3_lr', type=float, default=0.0002, help='0.01 for lcc | 0.00001 for AE | 0.0002 for GAN')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
parser.add_argument('--criticIters', type=int, default=1, help='for WGAN and WGAN-GP, number of critic iters per gen iter')
parser.add_argument('--Lh', type=float, default=1, help='LCCGAN-v2 hyperparameter')
parser.add_argument('--Lv', type=float, default=0.0001, help='LCCGAN-v2 hyperparameter')
parser.add_argument('--LCCLAMBDA', type=float, default=0.2, help='LCC hyperparameter')
parser.add_argument('--LAMBDA', type=float, default=10, help='gradient penalty lambda hyperparameter')

# model configuration
parser.add_argument('--ngfmin', type=int, default=16)
parser.add_argument('--ngfmax', type=int, default=512)
parser.add_argument('--ndfmin', type=int, default=16)
parser.add_argument('--ndfmax', type=int, default=512)
parser.add_argument('--flag_wn', type=bool, default=True)           # use of equalized-learning rate.
parser.add_argument('--flag_bn', type=bool, default=False)          # use of batch-normalization. (not recommended)
parser.add_argument('--flag_pixelwise', type=bool, default=True)    # use of pixelwise normalization for generator.
parser.add_argument('--flag_gdrop', type=bool, default=True)        # use of generalized dropout layer for discriminator.
parser.add_argument('--flag_leaky', type=bool, default=True)        # use of leaky relu instead of relu.
parser.add_argument('--flag_tanh', type=bool, default=True)        # use of tanh at the end of the generator.
parser.add_argument('--flag_sigmoid', type=bool, default=False)     # use of sigmoid at the end of the discriminator.
parser.add_argument('--flag_add_noise', type=bool, default=True)    # add noise to the real image(x)
parser.add_argument('--flag_norm_latent', type=bool, default=False) # pixelwise normalization of latent vector (z)
parser.add_argument('--flag_add_drift', type=bool, default=True)   # add drift loss
parser.add_argument('--ngf', type=int, default=64)             # feature dimension of final layer of generator.
parser.add_argument('--ndf', type=int, default=64)             # feature dimension of first layer of discriminator.

# configuration for progressive growing optimization
parser.add_argument('--niter', type=int, default=13, help='number of the first epoch to train for')
parser.add_argument('--trns_num', type=int, default=600000, help='number of fade in images')
parser.add_argument('--stab_num', type=int, default=600000, help='number of stablize images')
parser.add_argument('--niter1', type=int, default=100, help='number of stablize images')
parser.add_argument('--niter2', type=int, default=3, help='number of stablize images')
parser.add_argument('--niter3', type=int, default=100, help='number of stablize images')

# general settings
parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gpu', type=int, default=0, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--encoder', default='', help="path to netG (to continue training)")
parser.add_argument('--decoder', default='', help="path to netD (to continue training)")
parser.add_argument('--learnBasis', default='', help="path to learnBasis (to continue training)")
parser.add_argument('--lccGen', default='', help="path to netG (to continue training)")
parser.add_argument('--lccRec', default='', help="path to netG (to continue training)")
parser.add_argument('--outf', default='output_flowers', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', default=2, type=int, help='manual seed')

opt = parser.parse_args()
print(opt)