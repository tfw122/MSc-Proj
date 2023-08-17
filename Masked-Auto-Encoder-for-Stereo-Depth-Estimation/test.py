#from src.utils.builder import build_trainer, build_config
from src.utils.utils import *
from src.utils.builder import *
from src.utils.fileio import *
from arguments import args
from time import time
from src.testers.mil_tester import MILTester
from src.testers.mil_tester_internal import MILTesterInternal

def main():
    setup_imports()
    parser= args.get_parser()
    opts = parser.parse_args()

    config = build_config(opts)

    #fileio_client = FileIOClient(config)
    
    # load model; dataset etc;
    model = build_model(config, ckpt_path=config.model_config.load_checkpoint)
    print('model loaded with ckpt weights from \n {}'.format(config.model_config.load_checkpoint))
    
    data_module = build_datamodule(config)
    #data_module=None
    # start testing: config, data_module, model, run_name
    run_model=False
    tester = MILTesterInternal(config, data_module, model, run_model, 'internal_split_lv_v3_downstream')
    tester.test()
    

if __name__=='__main__':
    main()