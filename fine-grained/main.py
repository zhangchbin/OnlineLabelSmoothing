import torch
import torchvision
import argparse
from solver import Solver
from torchvision import transforms

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True


    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./data/CUB_200_2011/images/', help='CUB JPEGImages')
    parser.add_argument('--img_txt_train', type=str, default='./data/CUB_200_2011/train_list.txt', help='CUB train_list.txt')
    parser.add_argument('--img_txt_test', type=str, default='./data/CUB_200_2011/test_list.txt', help='CUB test_list.txt')
    parser.add_argument('--mode', type=str, default=None, help='train or test')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model')
    parser.add_argument('--load_model', type=str, default=None, help='load_model')
    parser.add_argument('--epochs', type=int, default=None, help='training epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning_rate')
    parser.add_argument('--resize_size', type=tuple, default=(224, 224), help='a tuple, resize the input image')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    
    parser.add_argument('--ensemble', nargs='+')

    ################
    parser.add_argument('--epochs_shift', type=int, default=1, help='in function validate_shift')
    parser.add_argument('--print_freq', type=int, default=1000, help='in validate function')
    parser.add_argument('--out_dir', type=str, default='./save_dir', help='in validation function')
    parser.add_argument('--arch', type=str, default=None, help='backbone')
    parser.add_argument('--dataset', type=str, default=None, help='cub, dogs, flowers, aircraft, cars')
    parser.add_argument('--method', type=str, default=None, help='baseline, ls, ols, kdols')
    parser.add_argument('--num_classes', type=int, default=None, help='the number of classes in dataset')
    parser.add_argument('--expname', type=str, default=None, help='exp name')
    ################
    args = parser.parse_args()
    
    if args.dataset == 'cub':
        args.img_path = './data/CUB_200_2011/images/'
        args.img_txt_train = './data/CUB_200_2011/lists/train_list.txt'
        args.img_txt_test  = './data/CUB_200_2011/lists/test_list.txt'
        args.num_classes = 200
    elif args.dataset == 'flowers':
        args.img_path = './data/flowers/jpg/'
        args.img_txt_train = './data/flowers/train_list.txt'
        args.img_txt_test  = './data/flowers/test_list.txt'
        args.num_classes = 102
    elif args.dataset == 'aircraft':
        args.img_path = './data/fgvc-aircraft-2013b/data/images/'
        args.img_txt_train = './data/fgvc-aircraft-2013b/trainval_list.txt'
        args.img_txt_test = './data/fgvc-aircraft-2013b/test_list.txt'
        args.num_classes = 90
    elif args.dataset == 'cars':
        args.img_path = './data/cars/car_ims/'
        args.img_txt_train = './data/cars/train_list.txt'
        args.img_txt_test = './data/cars/test_list.txt'
        args.num_classes = 196
    else:
        print('wrong dataset!!!!!!')

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    shift_validate_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    solver = Solver(args, train_transform=train_transform, test_transform=test_transform, val_transform = shift_validate_transform)
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'ensemble':
        solver.test_ensemble()


