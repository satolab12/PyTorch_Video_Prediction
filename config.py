import argparse

class ParseGRU():
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--dataset', default='/home/s/PycharmProjects/DATASET/car/train/',#avenue/training/frames/
                            help='log directory')
        parser.add_argument('--testset', default='/home/s/PycharmProjects/DATASET/car/test/',
                            help='log directory')

        parser.add_argument('--img_extension', default='jpg',
                            help='log directory')

        parser.add_argument('--num_itrs', type=int, default=10000)
        parser.add_argument('--T', type=int, default=4, help='checkpoint epoch')
        parser.add_argument('--num_layers', type=int, default=3, help='checkpoint epoch')
        parser.add_argument('--z_dim', type=int, default=100, help='weight decay')
        # parser.add_argument('--log_folder', default='./logs/actions/aegan', help='log directory')
        parser.add_argument('--log_folder', default='./save/images/', help='log directory')
        parser.add_argument('--batch_size', type=int,default=1)#../DATASET/UCSD/train/
        parser.add_argument('--test_batch', type=int, default=1)  # ../DATASET/UCSD/train6
        parser.add_argument('--sample_batch', type=int, default=1)  # ../DATASET/UCSD/train6
        parser.add_argument('--warmup', type=bool, default=False)  # ../DATASET/UCSD/train6
        parser.add_argument('--warmup_epochs', type=int, default=20)  # ../DATASET/UCSD/train6
        parser.add_argument('--image_size', default=128)
        parser.add_argument('--conv', default=2*2*2)  # img/16
        parser.add_argument('--check_point', type=int, default=1, help='apply SpectralNorm')#SNシますか?
        parser.add_argument('--n_test', type=int, default=1, help='apply Self-atten')  # Attnシますか?
        parser.add_argument('--n_channels', type=int, default=3, help='apply Self-atten')  # Attnシますか?
        parser.add_argument('--num_epochs', type=int, default=5, help='apply Self-atten')
        parser.add_argument('--gru_dim', type=int, default=128, help='weight decay')
        parser.add_argument('--ngru', type=int, default=100, help='dimension of latent variable')#512,128,32
        parser.add_argument('--alpha', type=int, default=25*4, help='weight decay')  # 1e-2,10
        parser.add_argument('--beta', type=float, default=2, help='weight decay')
        parser.add_argument('--lamda', type=int, default=10, help='weight decay')
        parser.add_argument('--try_', type=int, default=3, help='weight decay')

        #parser.add_argument('--learning_rate_d', type=int, default=6e-4, help='coefficient of L_prior')  # 1e-2
        parser.add_argument('--cuda', type=bool, default=True, help='weight decay')
        parser.add_argument('--learning_rate', type=float, default=2e-4, help='coefficient of L_prior')  # 1e-4,1e-3
        parser.add_argument('--learning_rate_d', type=float, default=2e-5, help='coefficient of L_prior')  # 4e-4,8e-3

        self.args = parser.parse_args()


