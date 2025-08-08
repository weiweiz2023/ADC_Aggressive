import argparse


def get_parser():

    parser = argparse.ArgumentParser(description='Hyperparameters for StoX-Net Training & Inference')

    ##################################################################################################################
    ## Regular Hyperparameters
    ##################################################################################################################
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--lr', type=float, default=0.01, 
                        help='learning rate (default: 0.01)')
    ##################################################################################################################
    ## High-level trainer parameters
    ##################################################################################################################
    parser.add_argument('--model', dest='model', default='resnet20', type=str, 
                        help='Choose a model to run the network on {resnet20, resnet_18}')
    parser.add_argument('--dataset', dest='dataset', help='Choose a dataset to run the network on from'
                                                          '{MNIST, CIFAR10}', default='MNIST', type=str)
    parser.add_argument('--experiment_state', default='pretraining', type=str, metavar='PATH',
                        help='What are we doing right now? options: [pretraining, pruning, PTQAT, inference, xbar_inference]')
    parser.add_argument('--run-info', default='', type=str,
                        help='Anything to add to the run name for clarification? e.g. \"test1ab\"')
   
    #parser.add_argument('--quantized', dest='quantized', default=False,
     #                   type=bool, help='Select whether use the quantized conv layer or not') 
    ##################################################################################################################
    ## Saving/Loading Data
    ##################################################################################################################
    parser.add_argument('--checkpoint-path', default='', type=str, metavar='PATH',
                        help='absolute path to desired checkpoint (default: none)')
    parser.add_argument('--model-save-dir', dest='model_save_dir',
                        help='The directory used to save the trained models',
                        default='./saved/models/', type=str)
    parser.add_argument('--logs-save-dir', dest='logs_save_dir',
                        help='The directory used to save the trained models',
                        default='./saved/logs/', type=str)
    parser.add_argument('--save-every', dest='save_every',
                        help='Saves checkpoints at every specified number of epochs',
                        type=int, default=10)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', type=str, default='/home/weiweiz/Documents/WW_02/ADC_aggressive/saved/models/_resnet20_MNIST_3_128_0p01_pruning_4_1_True_4_1_False_4_False_True_True_True_True_True_True.th', help='path to checkpoint to resume from')

    ##################################################################################################################
    ## QAT / Model Parameters
    ##################################################################################################################
    # xbar precision params
    parser.add_argument('--w_bits', default=0, type=int, metavar='N',
                        help='Number of weight bits (default: 1)')
    parser.add_argument('--w_bits_per_slice', default=1, type=int, metavar='N',
                        help='Number of weight bits per slice (default: 1), x <= 0 means full precision')
    parser.add_argument('--a_bits', default=0, type=int, metavar='N',
                        help='Number of input bits (default: 1)')
    parser.add_argument('--a_bits_per_stream', default=1, type=int, metavar='N',
                        help='Number of input bits per slice (default: 1), x <= 0 means full precision')
    parser.add_argument('--subarray-size', default=128, type=int, metavar='N',
                        help='Size of partial sum subarrays, x <= 0 means no partial sums')
    parser.add_argument('--slice-init', dest='slice_init', default=True,
                        type=bool, help='If W slices are present, create them at initialization of model')
    # adc params

    parser.add_argument('--Gon', default=1/10, type=int, metavar='N',
                        help='max conductance of ADC')
    parser.add_argument('--Goff', default=1/1000, type=int, metavar='N',
                        help='min conductance of ADC')
    parser.add_argument('--adc-prec', default=1, type=int, metavar='N',
                        help='ADC precision for quantized layers, x <= 0 means full precision')
    parser.add_argument('--save-adc', dest='save_adc', default=False,
                        type=bool, help='Select whether the ADC inputs are saved for analysis')
    parser.add_argument('--adc-grad-filter', dest='adc_grad_filter', default=True,
                        type=bool, help='Select whether an STE (False) or halfsine (True) is used for ADC backprop')
    parser.add_argument('--adc-round-method', dest='adc_stoch_round', default=False,
                        type=bool, help='Select whether stochaastic or deterministic rounding is used for ADC')
    parser.add_argument('--adc-pos-only', dest='adc_pos_only', default=False,
                        type=bool, help='Select whether [0, max] (True) or [-max, max-1] (False) is used')
    parser.add_argument('--adc-static-step', dest='adc_static_step', default=True,
                        type=bool, help='Select whether a static step size is used for ADC quant')
    parser.add_argument('--adc-custom-loss', dest='adc_custom_loss', default=False,
                        type=bool, help='Use custom loss additive term from ADC evals')
    parser.add_argument('--adc-shared', dest='shared_adc', default=True,
                        type=bool, help='Use a unique ADC for each subarray?')
    parser.add_argument('--Vmax', default=1, type=int,    
                        help='ADC Votalge')                    
    # other?
    parser.add_argument('--wa-stoch-round', dest='wa_stoch_round', default=True, 
                        help='Select whether stochaastic or deterministic rounding is used for ADC')
    parser.add_argument('--conv_prune_rate', default=0.6, type=float,
                        help='Set prune rate for')
    parser.add_argument('--linear_prune_rate', default=0.6, type=float,
                        help='Set prune rate for')
    parser.add_argument('--viz-comp-graph', default=False, type=bool, 
                        help='use torchviz to show model computational graph fwd/bkwd')
    ##################################################################################################################
    ## Other parameters
    ##################################################################################################################
    parser.add_argument('--print-batch-info', dest='print_batch_info',
                        help='Set to true if you want to see per batch accuracy',
                        default=False, type=bool)
    parser.add_argument('--skip-to-batch', default=0, type=int,
                        metavar='N', help='Skip to this batch of images in inference')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 20)')

    return parser
