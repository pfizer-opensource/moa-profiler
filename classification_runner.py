"""
Runner script for training, evaluation, and inference
"""
from classification import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="eval", help="train or eval or eval_compound_holdout")
    parser.add_argument("--distributed_data_parallel", default=False, action='store_true', help="whether to use DistributedDataParallel, if not present will use DataParallel by default")
    parser.add_argument("--study", type=str, default="JUMP1", help="JUMP1 or lincs")
    parser.add_argument("--label_type", type=str, default="moa_targets_compounds", help="if study is JUMP1, then one of [moa_targets_compounds, moa_targets_compounds_polycompound, moa_targets_compounds_holdout_2]. If study is lincs, then one of [moas_10uM, moas_10uM_polycompound, moas_10uM_compounds_holdout_2]")
    parser.add_argument("--eval_batch_size", type=int, default=30, help="batch size for evaluation")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs for training")
    parser.add_argument("--print_freq", type=int, default=1000, help="frequency of batches to print results")
    parser.add_argument("--lr", type=float, default=0.1, help="learning_rate")
    parser.add_argument("--wd", type=float, default=0.0001, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="learning momentum")
    parser.add_argument("--continue_training", default=False,  action='store_true', help="if mode=='train', whether to use pretrained model of task specified by label_type")
    parser.add_argument("--num_training_gpus", type=int,  default=4, help="if mode==train, number of gpus to use for training")
    parser.add_argument("--in_channels", type=int,  default=5, help="5 for standard model, 4 to skip the RNA channel")
    parser.add_argument("--class_aggregator", type=str, default="median", help="method by which to aggregate classes: median, pca")
    parser.add_argument("--well_aggregator", type=str, default="median", help="method by which to aggregate wells: median, pca")
    parser.add_argument("--load_latents", default=False,  action='store_true', help="if True, will load latent dictionaries from pickle instead of generating")
    parser.add_argument("--metric", type=str, default="pearson", help="similarity metric to use: pearson or cosine")
    
    opt = parser.parse_args()
    tagline=str(opt)
    print("parameters: ", opt)

    standardize = True
    if opt.study == "lincs":
        jitter = True
    else: ##JUMP1 image data type cannot be jittered with pytorch's transforms.ColorJitter
        jitter = False
    if opt.mode == "train":
        assert(torch.cuda.device_count() >= opt.num_training_gpus)
    else:
        assert(opt.continue_training==False)
   
    ##set batch sizes, train num workers, eval num workers; will actually have train_batch_size * number of gpus as the batch size being processed, this arg is the number per gpu; distributed data parallel will spawn a total number of workers = num_workers * number of processes (i.e. number of gpus)
    batch_worker_dict = {"JUMP1|distributed=True": (13,16,30) , "lincs|distributed=False": (55,30,30), "lincs|distributed=True":(14,16,30), "JUMP1|distributed=False":(50,30,30),  "combined|distributed=True":(14,16,30)}
    train_batch_size, train_num_workers, eval_num_workers = batch_worker_dict[opt.study + "|distributed=" + str(opt.distributed_data_parallel)]
    print("train_batch_size, train_num_workers, eval_num_workers: ", train_batch_size, train_num_workers, eval_num_workers)

    ##channel means and stds (after being scaled 0->1)
    if opt.study == "JUMP1" and (opt.label_type in ["gene_targets_compounds", "gene_targets_controls"] or "moa_targets_compounds" in opt.label_type):
        norm = pickle.load(open("pickles/JUMP1/channel_stats_compounds_raw.pkl", "rb"))
    if opt.study == "JUMP1" and opt.label_type == "gene_targets_all":
        norm = pickle.load(open("pickles/JUMP1/channel_stats_all_raw.pkl", "rb"))
    if opt.study == "lincs":
        norm = pickle.load(open("pickles/lincs/compressed_channel_stats_raw_corrected.pkl", "rb"))
    if opt.study == "combined":
        norm = pickle.load(open("pickles/combined/channel_stats.pkl", "rb"))   
    if opt.in_channels == 5:
        means = [norm["mean"][i] for i in range(0, 5)]
        stds = [norm["std"][i] for i in range(0, 5)]
    else:
        means = [norm["mean"][i] for i in range(0, 5) if i!=2] ##skip middle channel norm (RNA)
        stds = [norm["std"][i] for i in range(0, 5) if i!=2] 
        
    if standardize:
        data_transforms = transforms.Compose([transforms.Normalize(means, stds)])
    else:
        data_transforms = None

    ##size of latent representation 
    cardinality = 1280

    today = date.today().strftime("%b-%d-%Y")
    time = datetime.now().strftime("%H:%M:%S") 
    save_dir = "save_dir/{}/multiclass_classification/{}/".format(opt.study, today + "-" + time)

    csv_map = {
        "moa_targets_compounds": {"train": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_train.csv", "valid": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_valid.csv", "test": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test.csv", "test_no_neg": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test_no_negative.csv", "test_wells_excluded": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test_wells_excluded.csv", "full": "csvs/JUMP1/compound_images_with_MOA_no_polypharm_well_excluded.csv"},
        "moa_targets_compounds_polycompound": {"train": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_train.csv", "valid": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_valid.csv", "test": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test_wells_excluded_polycompound.csv", "test_no_neg": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test_wells_excluded_polycompound.csv", "test_wells_excluded": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test_wells_excluded_polycompound.csv", "full": "csvs/JUMP1/compound_images_with_MOA_no_polypharm_well_excluded.csv"},
        "moa_targets_compounds_holdout_2": {"train": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_well_excluded_compound_holdout_train_2_balanced_moas.csv", "valid": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_well_excluded_compound_holdout_valid_2.csv", "test": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_well_excluded_compound_holdout_test_2.csv",  "test_no_neg": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_well_excluded_compound_holdout_test_2_no_neg.csv", "test_wells_excluded": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_well_excluded_compound_holdout_test_2.csv", "full": "csvs/JUMP1/compound_images_with_MOA_no_polypharm_well_excluded.csv"},    
        "moa_targets_compounds_four_channel": {"train": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_train.csv", "valid": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_valid.csv", "test": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test.csv", "test_no_neg": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test_no_negative.csv", "test_wells_excluded": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test_wells_excluded.csv", "full": "csvs/JUMP1/compound_images_with_MOA_no_polypharm_well_excluded.csv"},
        
        "moas_10uM": {"train": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_train.csv", "valid": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_valid.csv", "test": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_test.csv", "test_no_neg": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_test_no_negative.csv", "full": "csvs/lincs/lincs_ten_micromolar_no_polypharm.csv"}, 
        "moas_10uM_polycompound": {"train": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_train.csv", "valid": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_valid.csv" , "test": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_test_polycompound.csv" , "test_no_neg": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_test_polycompound.csv", "full": "csvs/lincs/lincs_ten_micromolar_no_polypharm.csv"}, 
        "moas_10uM_compounds_holdout_2": {"train": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_compound_holdout_train_2_balanced_moas.csv" , "valid": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_compound_holdout_valid_2.csv", "test": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_compound_holdout_test_2.csv", "test_no_neg": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_compound_holdout_test_2_no_neg.csv", "full": "csvs/lincs/lincs_ten_micromolar_no_polypharm.csv"}, 
        "moas_10uM_four_channel": {"train": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_train.csv", "valid": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_valid.csv", "test": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_test.csv", "test_no_neg": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_test_no_negative.csv", "full": "csvs/lincs/lincs_ten_micromolar_no_polypharm.csv"}, 
    }

    if opt.label_type in ["moa_targets_compounds", "moa_targets_compounds_polycompound", "moa_targets_compounds_holdout_2"]:
        label_index_map = pickle.load(open("pickles/JUMP1/label_index_map_from_compound_images_with_MOA_no_polypharm.csv.pkl", "rb"))
        train_set = JUMPMOADataset(csv_map[opt.label_type]["train"], data_transforms, jitter=jitter, label_index_map=label_index_map, augment=True)
        valid_set = JUMPMOADataset(csv_map[opt.label_type]["valid"], data_transforms, jitter=False, label_index_map=label_index_map, augment=False)
        test_set = JUMPMOADataset(csv_map[opt.label_type]["test"], data_transforms, jitter=False, label_index_map=label_index_map, augment=False)
        test_set_no_negative = JUMPMOADataset(csv_map[opt.label_type]["test_no_neg"], data_transforms, jitter=False, label_index_map=label_index_map, augment=False)
        test_set_wells_excluded = JUMPMOADataset(csv_map[opt.label_type]["test_wells_excluded"], data_transforms, jitter=False, label_index_map=label_index_map, augment=False)
        full_dataset = JUMPMOADataset(csv_map[opt.label_type]["full"], data_transforms, jitter=False, label_index_map=label_index_map, augment=False)
    if opt.label_type == "moa_targets_compounds_holdout_2": 
        training_compounds = set(pd.read_csv(csv_map[opt.label_type]["train"])["perturbation"])
        test_set_wells_excluded = test_set ##test set here is already well-excluded
    if "moa_targets_compounds_replicates=" in opt.label_type:
        max_replicates = opt.label_type.split("=")[1]
        label_index_map = pickle.load(open("pickles/JUMP1/label_index_map_from_compound_images_with_MOA_no_polypharm.csv.pkl", "rb"))
        train_set = JUMPMOADataset("csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_replicates={}_train.csv".format(max_replicates), data_transforms, jitter=jitter, label_index_map=label_index_map, augment=True)    
        valid_set =JUMPMOADataset("csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_valid.csv", data_transforms, jitter=False, label_index_map=label_index_map, augment=False)    
        test_set = JUMPMOADataset("csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test.csv", data_transforms, jitter=False, label_index_map=label_index_map, augment=False)    
        test_set_no_negative = JUMPMOADataset("csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test_no_negative.csv", data_transforms, jitter=False, label_index_map=label_index_map, augment=False)    
        test_set_wells_excluded = JUMPMOADataset("csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test_wells_excluded.csv", data_transforms, jitter=False, label_index_map=label_index_map, augment=False)        
        full_dataset = JUMPMOADataset("csvs/JUMP1/compound_images_with_MOA_no_polypharm_well_excluded.csv", data_transforms, jitter=False, label_index_map=label_index_map, augment=False)
    if opt.label_type == "moa_targets_compounds_four_channel": 
        label_index_map = pickle.load(open("pickles/JUMP1/label_index_map_from_compound_images_with_MOA_no_polypharm.csv.pkl", "rb"))
        train_set = FourChannelJUMPClassificationDataset(csv_map[opt.label_type]["train"], data_transforms, jitter=jitter, label_index_map=label_index_map, augment=True)    
        valid_set =FourChannelJUMPClassificationDataset(csv_map[opt.label_type]["valid"], data_transforms, jitter=False, label_index_map=label_index_map, augment=False)    
        test_set = FourChannelJUMPClassificationDataset(csv_map[opt.label_type]["test"], data_transforms, jitter=False, label_index_map=label_index_map, augment=False)    
        test_set_no_negative = FourChannelJUMPClassificationDataset(csv_map[opt.label_type]["test_no_neg"], data_transforms, jitter=False, label_index_map=label_index_map, augment=False)    
        test_set_wells_excluded = FourChannelJUMPClassificationDataset(csv_map[opt.label_type]["test_wells_excluded"], data_transforms, jitter=False, label_index_map=label_index_map, augment=False)        
        full_dataset = FourChannelJUMPClassificationDataset(csv_map[opt.label_type]["full"], data_transforms, jitter=False, label_index_map=label_index_map, augment=False)
    
    if opt.label_type in ["moas_10uM", "moas_10uM_polycompound", "moas_10uM_compounds_holdout_2"]: 
        label_index_map = pickle.load(open("pickles/lincs/label_index_map_from_lincs_ten_micromolar_no_polypharm.csv.pkl", "rb"))
        train_set = LINCSClassificationDataset(csv_map[opt.label_type]["train"], data_transforms, jitter=jitter, label_index_map=label_index_map, augment=True)    
        valid_set = LINCSClassificationDataset(csv_map[opt.label_type]["valid"], data_transforms, jitter=False, label_index_map=label_index_map, augment=False)
        test_set = LINCSClassificationDataset(csv_map[opt.label_type]["test"], data_transforms, jitter=False, label_index_map=label_index_map, augment=False)
        test_set_no_negative = LINCSClassificationDataset(csv_map[opt.label_type]["test_no_neg"], data_transforms, jitter=False, label_index_map=label_index_map, augment=False)    
        full_dataset = LINCSClassificationDataset(csv_map[opt.label_type]["full"], data_transforms, label_index_map=label_index_map, augment=False)
    if opt.label_type == "moas_10uM_compounds_holdout_2": 
        training_compounds = set(pd.read_csv(csv_map[opt.label_type]["train"])["perturbation"])
    if "moas_10uM_replicates=" in opt.label_type: 
        max_replicates = opt.label_type.split("=")[1]
        label_index_map = pickle.load(open("pickles/lincs/label_index_map_from_lincs_ten_micromolar_no_polypharm.csv.pkl", "rb"))
        train_set = LINCSClassificationDataset("csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_replicates={}_train.csv".format(max_replicates), data_transforms, jitter=jitter, label_index_map=label_index_map, augment=True)    
        valid_set = LINCSClassificationDataset("csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_valid.csv", data_transforms, jitter=False, label_index_map=label_index_map, augment=False)
        test_set = LINCSClassificationDataset("csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_test.csv", data_transforms, jitter=False, label_index_map=label_index_map, augment=False)
        test_set_no_negative = LINCSClassificationDataset("csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_test_no_negative.csv", data_transforms, jitter=False, label_index_map=label_index_map, augment=False)    
        full_dataset = LINCSClassificationDataset("csvs/lincs/lincs_ten_micromolar_no_polypharm.csv", data_transforms, label_index_map=label_index_map, augment=False)
    if opt.label_type == "moas_10uM_four_channel": 
        label_index_map = pickle.load(open("pickles/lincs/label_index_map_from_lincs_ten_micromolar_no_polypharm.csv.pkl", "rb"))
        train_set = FourChannelLINCSClassificationDataset(csv_map[opt.label_type]["train"], data_transforms, jitter=jitter, label_index_map=label_index_map, augment=True)    
        valid_set = FourChannelLINCSClassificationDataset(csv_map[opt.label_type]["valid"], data_transforms, jitter=False, label_index_map=label_index_map, augment=False)
        test_set = FourChannelLINCSClassificationDataset(csv_map[opt.label_type]["test"], data_transforms, jitter=False, label_index_map=label_index_map, augment=False)
        test_set_no_negative = FourChannelLINCSClassificationDataset(csv_map[opt.label_type]["test_no_neg"], data_transforms, jitter=False, label_index_map=label_index_map, augment=False)    
        full_dataset = FourChannelLINCSClassificationDataset(csv_map[opt.label_type]["full"], data_transforms, label_index_map=label_index_map, augment=False)

    num_classes = len(label_index_map)

    ##instantiate data loaders and model
    if not opt.distributed_data_parallel: ##for distributedDataParallel need to instantiate data loader in the training method instead of globally
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=(torch.cuda.is_available()), num_workers=train_num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=opt.eval_batch_size, shuffle=False, pin_memory=(torch.cuda.is_available()), num_workers=eval_num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.eval_batch_size, shuffle=True, pin_memory=(torch.cuda.is_available()), num_workers=eval_num_workers)
    test_loader_no_negative = torch.utils.data.DataLoader(test_set_no_negative, batch_size=opt.eval_batch_size, shuffle=True, pin_memory=(torch.cuda.is_available()), num_workers=eval_num_workers)
    
    if opt.study == "JUMP1": ##JUMP1 dataset has problematic wells on plate BR00116995
        test_loader = torch.utils.data.DataLoader(test_set_wells_excluded, batch_size=opt.eval_batch_size, shuffle=True, pin_memory=(torch.cuda.is_available()), num_workers=eval_num_workers)
    full_dataset_loader = torch.utils.data.DataLoader(full_dataset, batch_size=opt.eval_batch_size, shuffle=True, pin_memory=(torch.cuda.is_available()), num_workers=eval_num_workers)
    
    model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes, in_channels=opt.in_channels)

    model_dictionary = {
        'JUMP1|moa_targets_compounds': 'save_dir/JUMP1/multiclass_classification/Jun-22-2022-08:49:11/models/model_best.dat',
        'JUMP1|moa_targets_compounds_polycompound': 'save_dir/JUMP1/multiclass_classification/Jun-22-2022-08:49:11/models/model_best.dat',
        'JUMP1|moa_targets_compounds_holdout_2': 'save_dir/JUMP1/multiclass_classification/Oct-18-2022-18:07:09/models/model_best.dat',

        'lincs|moas_10uM': 'save_dir/lincs/multiclass_classification/Jul-04-2022-15:00:03/models/model_best.dat',
        'lincs|moas_10uM_polycompound': 'save_dir/lincs/multiclass_classification/Jul-04-2022-15:00:03/models/model_best.dat',
        'lincs|moas_10uM_compounds_holdout_2': 'save_dir/lincs/multiclass_classification/Oct-23-2022-16:27:06/models/model_best.dat'
    } 

    if opt.continue_training or opt.mode == "eval" or opt.mode == "eval_compound_holdout":
        load_model = model_dictionary["{}|{}".format(opt.study, opt.label_type)]
        state_dict = torch.load(load_model, map_location="cpu") ##for continue training, load to CPU to avoid loading to GPU and getting device occupied error 
        new_state_dict = OrderedDict() ##state dict has prepending "module." as keys because DistributedDataParallel and DataParallel save with "module." in keynames, need to rename
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("loaded model: ", load_model)

    ##train
    if opt.mode == "train":
        if opt.distributed_data_parallel:
            nodes = 1 ##one machine, i.e. g###
            gpus = opt.num_training_gpus
            world_size = gpus * nodes
            nr = 0 
            gpu = 0 ##default gpu 0 as main ##specifiying device ID?
            master_addr = "0.0.0.0"
            master_port = findFreePort() 
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port
            makeSaveDir(save_dir, tagline) ##need to save now instead of in train method, we'll have mulitple processes trying to save, will error 
            args = (model, train_set, valid_loader, test_loader, save_dir, opt.n_epochs, train_batch_size, opt.lr, opt.wd, opt.momentum, None, opt.print_freq, tagline, world_size, nr, gpus, master_addr, master_port, train_num_workers)
            ##train the model - spawn method requires that first arg of input function be the gpu/process number
            tmp.spawn(train_distributed, nprocs=world_size, args=args)
        else: ##use DataParallel instead of DistributedDataParallel
            model=model.cuda()
            model = torch.nn.DataParallel(model, device_ids=list(range(0, opt.num_training_gpus)))
            ##train the model
            train(model=model, train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader, save=save_dir, n_epochs=opt.n_epochs, batch_size=train_batch_size, lr=opt.lr, wd=opt.wd, momentum=opt.momentum, seed=None, print_freq=opt.print_freq, tagline=tagline)
            model = model.module ##go from DataParallel back to model instance in preparation for model evaluation
    
    if opt.mode == "eval" or opt.mode == "eval_compound_holdout": ##get CP and DP latent dictionaries  
        aggregate_by_well = True
        drop_neg_control = True
        ##CellProfiler and DeepProfiler analysis
        for method in ["cellProfiler", "deepProfiler"]:
            print("{} {}:".format(opt.study, method))
            if method == "deepProfiler":
                print("DP extraction and analysis")
                if opt.study == "JUMP1":
                    deep_profile_types = ["model_{}".format(opt.well_aggregator)]
                else:
                    deep_profile_types = ["model_{}".format(opt.well_aggregator), "bornholdt_trained"]
                for deep_profile_type in deep_profile_types:
                    if opt.load_latents:
                        print("loaded latent dictionary: {} deep profiler reps {} ".format(opt.study, deep_profile_type))
                        latent_dictionary = pickle.load(open("pickles/{}/plot/DP_latent_dictionary_label_type_{}_{}_full_dataset.pkl".format(opt.study, opt.label_type, deep_profile_type), "rb"))
                    else:
                        print("extracting {} deep profiler reps {} ".format(opt.study, deep_profile_type))
                        latent_dictionary = extractProfilerRepresentations(study=opt.study, method=method, loader=full_dataset_loader, deep_profile_type=deep_profile_type)
                        pickle.dump(latent_dictionary, open("pickles/{}/plot/DP_latent_dictionary_label_type_{}_{}_full_dataset.pkl".format(opt.study, opt.label_type, deep_profile_type), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
                    latent_dictionary = standardizeEmbeddings(latent_dictionary)

                    if opt.mode == "eval":
                        logistic_regression_scores = logisticRegression(latent_dictionary, study=opt.study, training_csv=csv_map[opt.label_type]["train"], test_csv=csv_map[opt.label_type]["test"], drop_neg_control=False) ##should use the full latent dictionary 
                        pickle.dump(logistic_regression_scores, open("pickles/{}/plot/logistic_regression_{}_CP_False_DP_True_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(opt.study, opt.class_aggregator, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well, deep_profile_type), "wb"))

                        latent_dictionary = filterToTestSet(latent_dictionary, csv_map[opt.label_type]["test"], study=opt.study)

                        accuracy_map, k_pred_labels_map, k_stats, replicate_correlation_map = getNeighborAccuracy(latent_dictionary, metric=opt.metric, verbose=True, drop_neg_control=drop_neg_control)
                        pickle.dump(accuracy_map, open("pickles/{}/plot/kNN_map_CP_False_DP_True_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(opt.study, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well, deep_profile_type), "wb"))
                        pickle.dump(k_pred_labels_map, open("pickles/{}/plot/k_pred_map_CP_False_DP_True_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(opt.study, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well, deep_profile_type), "wb"))
                        pickle.dump(k_stats, open("pickles/{}/plot/k_stats_CP_False_DP_True_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(opt.study, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well, deep_profile_type), "wb"))
                        pickle.dump(replicate_correlation_map, open("pickles/{}/plot/replicate_correlation_CP_False_DP_True_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(opt.study, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well, deep_profile_type), "wb"))

                        pred_labels_map = getAccuracyByClosestAggregateLatent(latent_dictionary, class_aggregator=opt.class_aggregator, metric=opt.metric, drop_neg_control=drop_neg_control)
                        pickle.dump(pred_labels_map, open("pickles/{}/plot/pred_labels_map_{}_latent_CP_False_DP_True_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(opt.study, opt.class_aggregator, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well, deep_profile_type), "wb"))

                        enrichment_dict = getEnrichment(latent_dictionary, metric=opt.metric, drop_neg_control=drop_neg_control)
                        pickle.dump(enrichment_dict, open("pickles/{}/plot/enrichment_{}_latent_CP_False_DP_True_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(opt.study, opt.class_aggregator, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well, deep_profile_type), "wb"))

                    ##compound holdout latent 
                    if opt.mode == "eval_compound_holdout":
                        pred_labels_map, k_map, latent_k_map = compoundHoldoutClassLatentAssignment(latent_dictionary, study=opt.study, class_aggregator=opt.class_aggregator, metric=opt.metric, drop_neg_control=drop_neg_control, training_compounds=training_compounds, label_index_map=label_index_map)
                        pickle.dump(pred_labels_map, open("pickles/{}/plot/pred_labels_map_{}_latent_CP_False_DP_True_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(opt.study, opt.class_aggregator, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well, opt.well_aggregator), "wb"))
                        pickle.dump(k_map, open("pickles/{}/plot/latent_vote_by_wells_k_map_{}_False_True_{}_{}_{}_{}_{}.pkl".format(opt.study, opt.class_aggregator, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well, deep_profile_type), "wb"))
                        pickle.dump(latent_k_map, open("pickles/{}/plot/latent_vote_by_embedding_{}_False_True_{}_{}_{}_{}_{}.pkl".format(opt.study, opt.class_aggregator, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well, deep_profile_type), "wb"))
            
            else: ##CP analysis
                if opt.load_latents:
                    latent_dictionary = pickle.load(open("pickles/{}/plot/CP_latent_dictionary_label_type_{}_full_dataset.pkl".format(opt.study, opt.label_type), "rb"))
                    print("loaded CP latent")
                else:
                    print("CP extraction and analysis")
                    latent_dictionary = extractProfilerRepresentations(study=opt.study, method=method, loader=full_dataset_loader)
                    pickle.dump(latent_dictionary, open("pickles/{}/plot/CP_latent_dictionary_label_type_{}_full_dataset.pkl".format(opt.study, opt.label_type), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
                latent_dictionary = standardizeEmbeddings(latent_dictionary)
            
                if opt.mode == "eval":
                    logistic_regression_scores = logisticRegression(latent_dictionary, study=opt.study, training_csv=csv_map[opt.label_type]["train"], test_csv=csv_map[opt.label_type]["test"], drop_neg_control=False)
                    pickle.dump(logistic_regression_scores, open("pickles/{}/plot/logistic_regression_{}_CP_True_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}.pkl".format(opt.study, opt.class_aggregator, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well), "wb"))

                    latent_dictionary = filterToTestSet(latent_dictionary, csv_map[opt.label_type]["test"], study=opt.study)

                    accuracy_map, k_pred_labels_map, k_stats, replicate_correlation_map = getNeighborAccuracy(latent_dictionary, metric=opt.metric, verbose=True, drop_neg_control=drop_neg_control)
                    pickle.dump(accuracy_map, open("pickles/{}/plot/kNN_map_CP_True_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}.pkl".format(opt.study, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well), "wb"))
                    pickle.dump(k_pred_labels_map, open("pickles/{}/plot/k_pred_map_CP_True_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}.pkl".format(opt.study, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well), "wb"))
                    pickle.dump(k_stats, open("pickles/{}/plot/k_stats_CP_True_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}.pkl".format(opt.study, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well), "wb"))
                    pickle.dump(replicate_correlation_map, open("pickles/{}/plot/replicate_correlation_CP_True_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}.pkl".format(opt.study, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well), "wb"))
                
                    pred_labels_map = getAccuracyByClosestAggregateLatent(latent_dictionary, class_aggregator=opt.class_aggregator, metric=opt.metric, drop_neg_control=drop_neg_control)
                    pickle.dump(pred_labels_map, open("pickles/{}/plot/pred_labels_map_{}_latent_CP_True_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}.pkl".format(opt.study, opt.class_aggregator, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well), "wb"))

                    enrichment_dict = getEnrichment(latent_dictionary, metric=opt.metric, drop_neg_control=drop_neg_control)
                    pickle.dump(enrichment_dict, open("pickles/{}/plot/enrichment_{}_latent_CP_True_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}.pkl".format(opt.study, opt.class_aggregator, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well), "wb"))

                ##compound holdout latent 
                if opt.mode == "eval_compound_holdout":                    
                    pred_labels_map, k_map, latent_k_map = compoundHoldoutClassLatentAssignment(latent_dictionary, study=opt.study, class_aggregator=opt.class_aggregator, metric=opt.metric, drop_neg_control=drop_neg_control, training_compounds=training_compounds, label_index_map=label_index_map)
                    pickle.dump(pred_labels_map, open("pickles/{}/plot/pred_labels_map_{}_latent_CP_True_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(opt.study, opt.class_aggregator, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well, opt.well_aggregator), "wb"))
                    pickle.dump(k_map, open("pickles/{}/plot/latent_vote_by_wells_k_map_{}_True_False_{}_{}_{}_{}_None.pkl".format(opt.study, opt.class_aggregator, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well), "wb"))
                    pickle.dump(latent_k_map, open("pickles/{}/plot/latent_vote_by_embedding_{}_True_False_{}_{}_{}_{}_None.pkl".format(opt.study, opt.class_aggregator, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well), "wb"))


    ##generate MP latent dictionary
    if opt.mode == "eval" or opt.mode == "eval_compound_holdout":
        model=model.cuda()
        ##use the full dataset loader
        if opt.load_latents:
            latent_dictionary = pickle.load(open("pickles/{}/plot/latent_dictionary_label_type_{}_well_aggregated_True_{}_full_dataset.pkl".format(opt.study, opt.label_type, opt.well_aggregator), "rb"))
        else:
            latent_dictionary = getLatentRepresentations(study=opt.study, well_aggregator=opt.well_aggregator, model=model, loader=full_dataset_loader, cardinality=cardinality, label_index_map=label_index_map)
            pickle.dump(latent_dictionary, open("pickles/{}/plot/latent_dictionary_label_type_{}_well_aggregated_True_{}_full_dataset.pkl".format(opt.study, opt.label_type, opt.well_aggregator), "wb"), protocol=pickle.HIGHEST_PROTOCOL)        

    if opt.mode == "eval": ##evaluate and run normal inferences
        logistic_regression_scores = logisticRegression(latent_dictionary, study=opt.study, training_csv=csv_map[opt.label_type]["train"], test_csv=csv_map[opt.label_type]["test"], drop_neg_control=False)
        pickle.dump(logistic_regression_scores, open("pickles/{}/plot/logistic_regression_{}_CP_False_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(opt.study, opt.class_aggregator, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well, opt.well_aggregator), "wb"))

        # ##classification accuracy reporting -- exclude negative DMSO control 
        classification_map = getClassificationStats(model=model, loader=test_loader_no_negative, label_index_map=label_index_map)
        pickle.dump(classification_map, open("pickles/{}/plot/classification_map_{}".format(opt.study, opt.label_type), "wb"))
        for stratify_by in ["plate_MOA", "plate", "well"]:
            stratified_performance_dict = getStratifiedPerformance(model=model, loader=test_loader,  stratify_by=stratify_by, label_index_map=label_index_map)
            pickle.dump(stratified_performance_dict, open("pickles/{}/plot/{}_specific_performance_map_{}.pkl".format(opt.study, stratify_by, opt.label_type), "wb"))
        ##latent analysis
        ##for evaluation, just use test set 
        latent_dictionary = filterToTestSet(latent_dictionary, csv_map[opt.label_type]["test"], study=opt.study)
        aggregate_by_well = True
        drop_neg_control = True
        enrichment_dict = getEnrichment(latent_dictionary,  metric=opt.metric, drop_neg_control=drop_neg_control)
        pickle.dump(enrichment_dict, open("pickles/{}/plot/enrichment_{}_latent_CP_False_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(opt.study, opt.class_aggregator, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well, opt.well_aggregator), "wb"))

        accuracy_map, k_pred_labels_map, k_stats, replicate_correlation_map = getNeighborAccuracy(latent_dictionary, metric=opt.metric, verbose=True, drop_neg_control=drop_neg_control)
        pickle.dump(accuracy_map, open("pickles/{}/plot/kNN_map_CP_False_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(opt.study, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well, opt.well_aggregator), "wb"))
        pickle.dump(k_pred_labels_map, open("pickles/{}/plot/k_pred_map_CP_False_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(opt.study, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well, opt.well_aggregator), "wb"))
        pickle.dump(k_stats, open("pickles/{}/plot/k_stats_CP_False_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(opt.study, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well, opt.well_aggregator), "wb"))
        pickle.dump(replicate_correlation_map, open("pickles/{}/plot/replicate_correlation_CP_False_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(opt.study, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well, opt.well_aggregator), "wb"))

        pred_labels_map = getAccuracyByClosestAggregateLatent(latent_dictionary, class_aggregator=opt.class_aggregator, metric=opt.metric, drop_neg_control=drop_neg_control)
        pickle.dump(pred_labels_map, open("pickles/{}/plot/pred_labels_map_{}_latent_CP_False_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(opt.study, opt.class_aggregator, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well, opt.well_aggregator), "wb"))

    if opt.mode == "eval_compound_holdout":
        ##load the full latent dictionary for compound holdout 
        latent_dictionary = pickle.load(open("pickles/{}/plot/latent_dictionary_label_type_{}_well_aggregated_True_{}_full_dataset.pkl".format(opt.study, opt.label_type, opt.well_aggregator), "rb"))
        model=model.cuda()
        pred_labels_map, k_map, latent_k_map = compoundHoldoutClassLatentAssignment(latent_dictionary, study=opt.study, class_aggregator=opt.class_aggregator, metric=opt.metric, drop_neg_control=drop_neg_control, training_compounds=training_compounds, label_index_map=label_index_map)
        pickle.dump(pred_labels_map, open("pickles/{}/plot/pred_labels_map_{}_latent_CP_False_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(opt.study, opt.class_aggregator, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well, opt.well_aggregator), "wb"))
        pickle.dump(k_map, open("pickles/{}/plot/latent_vote_by_wells_k_map_{}_False_False_{}_{}_{}_{}_None.pkl".format(opt.study, opt.class_aggregator, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well), "wb"))
        pickle.dump(latent_k_map, open("pickles/{}/plot/latent_vote_by_embedding_{}_False_False_{}_{}_{}_{}_None.pkl".format(opt.study, opt.class_aggregator, opt.metric, opt.label_type, drop_neg_control, aggregate_by_well), "wb"))

        img_k_map, field_k_map, corrects = getHoldoutCompoundPrediction(model=model, study=opt.study, loader=test_loader_no_negative, label_index_map=label_index_map)
        pickle.dump(field_k_map, open("pickles/{}/plot/compound_holdout_model_pred_well_k_map_{}.pkl".format(opt.study, opt.label_type), "wb"))
        pickle.dump(img_k_map, open("pickles/{}/plot/compound_holdout_model_pred_image_k_map_{}.pkl".format(opt.study, opt.label_type), "wb"))
        pickle.dump(corrects, open("pickles/{}/plot/compound_holdout_correctly_predicted_{}.pkl".format(opt.study, opt.label_type), "wb"))

if __name__ == "__main__":
    main()
