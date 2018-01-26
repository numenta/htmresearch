

def parse_argv():
    parser.add_option("--path",   type=str, default='', dest="path", help="...")
    parser.add_option("--metric", type=str, default='', dest="metric", help="...")
    parser.add_option("-e", type=int, default='', dest="num_epochs", help="...")
    (options, remainder) = parser.parse_args()
    return options, remainder

####################################################
####################################################
####################################################
### 
###                   Main()
### 
####################################################
####################################################
####################################################

def main(argv):
    args, _ = parse_argv()

    path       = args.path
    metric     = args.metric
    num_epochs = args.num_epochs

    

