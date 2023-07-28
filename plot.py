"""
Script for generating plots
"""
from classification import *

parser = argparse.ArgumentParser()
parser.add_argument("--class_aggregator", type=str, default="median", help="median or pca")
parser.add_argument("--well_aggregator", type=str, default="median", help="median or pca")
parser.add_argument("--metric", type=str, default="pearson", help="pearson or cosine")

opt = parser.parse_args()

def plotLossCurves(path, plt_type="loss"):
    """
    Plots loss curve based on CSV PATH, which must contain headers epoch, train_loss, train_error, valid_loss, valid_error
    Writes graph to same directory as PATH
    """
    df = pd.read_csv(path)
    train_loss = []
    train_error = []
    val_loss = []
    val_error = []
    epochs = []
    for index, row in df.iterrows():
        train_loss.append(row["train_loss"])
        train_error.append(row["train_error"])
        val_loss.append(row["valid_loss"])
        val_error.append(row["valid_error"])
        epochs.append(row["epoch"])
    fig, ax = plt.subplots()
    if plt_type == "loss":
        ax.plot(epochs, train_loss, label="train loss")
        ax.plot(epochs, val_loss, label="valid loss")
    if plt_type == "error":
        ax.set_ylim((0,1.03))
        ax.plot(epochs, train_error, label="train error")
        ax.plot(epochs, val_error, label="valid error")
    study = "JUMP1" if "JUMP1" in path else "LINCS"
    plt.title("{} {} Curves".format(study, plt_type.capitalize()), fontsize=14)
    ax.set_ylabel("{}".format(plt_type.capitalize()),fontsize=12)
    ax.set_xlabel("Epoch".format(plt_type.capitalize()),fontsize=12)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', prop={"size":10}, bbox_to_anchor=(1, 1.32))
    plt.gcf().subplots_adjust(top=.76)
    plt.savefig(path.replace("results.csv", "{}_curve.png".format(plt_type)), dpi=300)

def plotReplicateAndNonreplicateSimilarity(study=None, metric=None, label_type=None, drop_neg_control=None, aggregate_by_well=None, well_aggregator="mean", deep_profile_type=None):
    """
    Plots MOA-replicate vs non-MOA-replicate similarities
    """
    efficient_mapp = pickle.load(open("pickles/{}/plot/replicate_correlation_CP_False_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(study, metric, label_type, drop_neg_control, aggregate_by_well, well_aggregator), "rb"))
    cell_profiler_mapp = pickle.load(open("pickles/{}/plot/replicate_correlation_CP_True_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}.pkl".format(study, metric, label_type, drop_neg_control, aggregate_by_well), "rb"))
    deep_profiler_mapp = pickle.load(open("pickles/{}/plot/replicate_correlation_CP_False_DP_True_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(study, metric, label_type, drop_neg_control, aggregate_by_well, deep_profile_type), "rb"))
    keys = ["CellProfiler", "DeepProfiler", "MOAProfiler"]
    color_map = {"CellProfiler":"red", "DeepProfiler":"purple", "MOAProfiler":"gold"}
    correlation_map = {key: {"replicate":"", "nonreplicate":"", "Delta":""} for key in keys}
    for key in correlation_map:
        if key == "CellProfiler":
            mapp = cell_profiler_mapp
        if key == "DeepProfiler":
            mapp = deep_profiler_mapp
        if key == "MOAProfiler":
            mapp = efficient_mapp
        correlation_map[key]["replicate"] = mapp["replicate"]
        correlation_map[key]["nonreplicate"] = mapp["nonreplicate"]
        correlation_map[key]["Delta"] = (mapp["replicate"][0] - mapp["nonreplicate"][0], 0) 
    fig, ax = plt.subplots()
    xlabels = ["replicate", "nonreplicate", "Delta"]
    x = np.array([1,2,3])
    ax.set_xticks(x)
    width = .20
    for key in keys:
        similarities = [correlation_map[key][xlabel][0] for xlabel in xlabels]
        similarities_stds = [correlation_map[key][xlabel][1] for xlabel in xlabels]
        ax.bar(x, similarities, yerr=similarities_stds, capsize=3, width=width, color=color_map[key], label=key)
        for i,j in zip(x, similarities):
            ax.annotate("{:.0%}".format(j), xy=(i - .10, j +.03),fontsize=8)
        x = x + width
    if metric == "pearson":
        ax.set_ylabel("Pearson Correlation Coefficient")
        plt.title("{}: Intra-MOA vs Inter-MOA PCC Similarity".format(study.upper()))
    if metric == "cosine":
        ax.set_ylabel("Cosine Similarity")
        plt.title("{}: Intra-MOA vs Inter-MOA Cosine Similarity".format(study.upper()))
    xlabels = ["Intra-MOA", "Inter-MOA", "Delta"]
    ax.set_xticklabels(xlabels)
    ax.set_ylim((0, 1.08))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', prop={"size":9}, bbox_to_anchor=(1, 1.32))
    plt.gcf().subplots_adjust(top=.76)
    plt.savefig("outputs/replicateNonreplicateSimilarity_{}_{}_{}_drop_neg_{}_well_aggregated_{}_{}_{}.png".format(study, metric, label_type, drop_neg_control, aggregate_by_well, well_aggregator, deep_profile_type), dpi=300)

def plotScoreByAggregatedLatentRep(study=None, class_aggregator="mean", metric=None, label_type=None, drop_neg_control=None, aggregate_by_well=None, well_aggregator="mean", deep_profile_type=None):
    """
    Plots F1, precision, and recall
    """
    cell_profiler_dict = pickle.load(open("pickles/{}/plot/pred_labels_map_{}_latent_CP_True_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}.pkl".format(study, class_aggregator, metric, label_type, drop_neg_control, aggregate_by_well), "rb"))
    deep_profiler_dict = pickle.load(open("pickles/{}/plot/pred_labels_map_{}_latent_CP_False_DP_True_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(study, class_aggregator, metric, label_type, drop_neg_control, aggregate_by_well, deep_profile_type), "rb"))
    efficient_dict = pickle.load(open("pickles/{}/plot/pred_labels_map_{}_latent_CP_False_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(study, class_aggregator, metric, label_type, drop_neg_control, aggregate_by_well, well_aggregator), "rb"))
    maps = {"CellProfiler": cell_profiler_dict, "DeepProfiler": deep_profiler_dict, "MOAProfiler":efficient_dict}
    scores_map = {"F1": [], "Precision":[], "Recall":[]}
    color_map = {"F1": "grey", "Precision":"salmon", "Recall":"lightsteelblue"}
    xlabels = sorted(list(maps.keys())) 
    for m in xlabels:
        mapp = maps[m]
        scores_map["Precision"].append(sklearn.metrics.precision_score(mapp["labels"], mapp["predictions"], average="weighted", zero_division=0))
        scores_map["Recall"].append(sklearn.metrics.recall_score(mapp["labels"], mapp["predictions"], average="weighted", zero_division=0))
        scores_map["F1"].append(sklearn.metrics.f1_score(mapp["labels"], mapp["predictions"], average="weighted", zero_division=0))
    fig, ax = plt.subplots()
    width = .20
    x = np.array((range(1, len(xlabels) + 1)))
    ax.set_xticks(x)
    for score_type in scores_map:
        scores = scores_map[score_type]
        print("{} {} class latent {} {} , MP percent improvement over CP: {} and DP: {}".format(study, label_type, score_type, scores, (scores[2] - scores[0]) / scores[0], (scores[2] - scores[1]) / scores[1]))
        bar = ax.bar(x, scores, width=width, color=color_map[score_type], label=score_type)
        for i,j in zip(x, scores):
            ax.annotate("{:.0%}".format(j), xy=(i - .03, j +.03),fontsize=6)
        x = x + width 
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Score")
    ax.set_ylim((0,1.03))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', prop={"size":9}, bbox_to_anchor=(1, 1.32))
    plt.gcf().subplots_adjust(top=.76)
    plt.title("{}: Class Latent Assignment Metrics".format(study.upper()))
    plt.savefig("outputs/scores_by_{}_latent_rep_{}_{}_{}_drop_neg_{}_well_aggregated_{}_{}_{}.png".format(class_aggregator, study, metric, label_type, drop_neg_control, aggregate_by_well, well_aggregator, deep_profile_type), dpi=300)

def getDMSOWellLocations(study="JUMP1"):
    """
    Helper function to get the well locations as a set
    """
    if study == "JUMP1":
        df = pd.read_csv("csvs/JUMP1/compound_images_with_MOA_no_polypharm.csv")
    if study == "lincs":
        df = pd.read_csv("csvs/lincs/lincs_ten_micromolar_no_polypharm.csv")
    wells = set()
    for index, row in df.iterrows():
        if row["perturbation"] in ["Empty", "DMSO"]:
            wells.add(getRowColumn(row["imagename"]))
    return wells

def plotCellTypeTimepointSpecificPerformance(study="JUMP1", label_type=None):
    if study != "JUMP1":
        return ##only JUMP1 has cell type and timepoint differentiation
    for strat in ["cell_type", "timepoint"]:
        fig, ax = plt.subplots()
        stratified_performance_dict = pickle.load(open("pickles/{}/plot/{}_specific_performance_map_{}.pkl".format(study, strat, label_type), "rb"))   
        xlabels = sorted(list(stratified_performance_dict.keys()))
        x = list(range(1, len(xlabels) + 1))
        y = [stratified_performance_dict[key][0] for key in xlabels]
        n = [stratified_performance_dict[key][1] for key in xlabels]
        ax.bar(x, y)
        ax.set_xticks(x)
        ax.set_ylim(0, 1)
        if strat == "cell_type":
            xlabels = [str(xlabels[i]) + "\nn={} images".format(n[i]) for i in range(0, len(n))]
        else:
            xlabels = [str(xlabels[i]) + " Hours\nn={} images".format(n[i]) for i in range(0, len(n))]
        ax.set_xticklabels(xlabels)
        ax.set_ylabel("Accuracy")
        for i,j in zip(x, y):
            ax.annotate("{:.0%}".format(j), xy=(i - .05, j +.02),fontsize=8)
        plt.title("JUMP1 Classification Performance by {}".format(strat.replace("_", " ").title()))
        plt.savefig("outputs/{}_specific_performance_{}_{}.png".format(strat, study, label_type), dpi=300)
    
def plotWellSpecificPerformance(study="JUMP1", label_type=None):
    """
    Plots plate classification heatmap
    """
    stratified_performance_dict = pickle.load(open("pickles/{}/plot/well_specific_performance_map_{}.pkl".format(study, label_type), "rb"))   
    DMSO_wells = getDMSOWellLocations(study=study)
    ##convert to 2D array and display 
    plate = [] ##matrix of classification accuracies
    sample_sizes = [] ##matrix of sample sizes
    for r in range(1,17):
        row = []
        row_sample_sizes = []
        if r < 10:
            string_r = "0" + str(r)
        else:
            string_r = str(r)
        for c in range(1, 25):
            if c < 10:
                string_c = "0" + str(c)
            else:
                string_c = str(c)
            key = 'r' + string_r + 'c' + string_c
            if key in stratified_performance_dict:
                row.append(stratified_performance_dict['r'+string_r + 'c' + string_c][0])
                row_sample_sizes.append(stratified_performance_dict['r'+string_r + 'c' + string_c][1])
            else:
                row.append(-1)
                row_sample_sizes.append(-1)
        plate.append(row)
        sample_sizes.append(row_sample_sizes)
    plate=np.array(plate)
    sample_sizes=np.array(sample_sizes)
    plotPlateEdgeStatistics(plate, sample_sizes, plate_label="classification_accuracy", study=study, label_type=label_type)
    ##plot it
    fig, ax = plt.subplots()
    im = ax.imshow(plate, vmin=0.0, vmax=1.0)
    ##annotate DMSO wells
    DMSO_annotated = set()
    for well in DMSO_wells:
        i = int(well[well.find("r") + 1: well.find("c")]) - 1
        j = int(well[well.find("c") + 1: ]) - 1 
        text = ax.text(j, i, "D",  ha="center", va="center", color="white", fontsize=6)
        DMSO_annotated.add((i,j))
    ##annotate missing wells (if any)
    for i in range(0, len(plate)):
        for j in range(0, len(plate[i])):
            if plate[i][j] == -1 and (i,j) not in DMSO_annotated:
                text = ax.text(j, i, "N/A",  ha="center", va="center", color="grey", fontsize=6)
    ax.set_xticks(np.arange(24))
    ax.set_yticks(np.arange(16))
    ax.set_xticklabels(list(range(1, 25)),fontsize=8)
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
    ax.set_yticklabels(letters,fontsize=8)
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.title("{}: Classification Accuracy\nby Well Location".format(study.upper()))
    plt.savefig("outputs/platemap_accuracy_{}_{}.pkl.png".format(study, label_type), dpi=300)

def plotPlateEdgeStatistics(plate, sample_sizes, plate_label=None, study=None, label_type=None, well_aggregator=None):
    """
    Given a 16 x 24 = 384 well PLATE matrix, calculates p-value of edge mean compared to non-edge mean and plots bar graph
    """
    edges = [] #80 - 4 = 76
    non_edges = [] #384 - 76 = 308 (not counting the N/As)
    nas = 0
    for i in range(0, len(plate)): ##plate is 16x24
        for j in range(0, len(plate[i])):
            if plate[i][j] == -1: ##skip the N/A wells
                nas += 1
                continue
            ##plate[i][j] is an average, sample_sizes[i][j] is the count, need to break down to individual samples of correct classification (one) and incorrect (zero)
            ones = int(plate[i][j] * sample_sizes[i][j]) 
            zeros = int(sample_sizes[i][j] - ones)
            assert(ones + zeros == sample_sizes[i][j])
            if i == 0 or i == 15 or j == 0 or j == 23: ##if edge
                edges += [1] * ones 
                edges += [0] * zeros
            else:
                non_edges += [1] * ones 
                non_edges += [0] * zeros
    edge_n, non_edge_n = len(edges), len(non_edges)
    edge_mean, edge_std = np.mean(edges), np.std(edges)
    non_edge_mean, non_edge_std = np.mean(non_edges), np.std(non_edges)
    z_val = (edge_mean - non_edge_mean) / float( np.sqrt( ((edge_std**2) / float(len(edges))) + ((non_edge_std**2) / float(len(non_edges))) )) 
    cdf_one_sided = scipy.stats.norm.cdf(z_val) 
    cdf_two_sided = (scipy.stats.norm.cdf(z_val) * 2) - 1 
    p_val_one_sided = round(1 - cdf_one_sided, 2) ##we want the one sided with Ho: means are equal, alternative: mean greater
    p_val_two_sided = round(1 - cdf_two_sided, 2) ##we want the one sided with Ho: means are equal, alternative: means are not equal 
    fig, ax = plt.subplots()
    xlabels = ["Edge", "Non-Edge"]
    x = np.array([1,2])
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    if plate_label=="classification_accuracy":
        ax.set_ylabel("Classification Accuracy")
        plt.title("{}: Edge vs Non-Edge Average Classification Accuracy".format(study.upper()))    
    if plate_label == "well_correlations":
        ax.set_ylabel("Well Pearson Correlations")
        plt.title("{}: Edge vs Non-Edge Average Well Pearson Correlations".format(study.upper()))        
    width = .4
    bar = ax.bar(x, [edge_mean, non_edge_mean], yerr=[edge_std, non_edge_std], color=("blue", "orange"))
    for i,j in zip(x, [(edge_mean, edge_n), (non_edge_mean, non_edge_n)]):
        ax.annotate("{:.0%}\nn={}".format(j[0], j[1]), xy=(i - .22, j[0] +.03),fontsize=8)
        x = x + width
    plt.savefig("outputs/edge_vs_nonedge_{}_{}_{}_{}.png".format(study, plate_label, label_type, well_aggregator), dpi=300)

def plotClassificationMetrics(study, label_type):
    """
    Plots PRC and accuracy of classifier
    """
    efficient_classification_map = pickle.load(open("pickles/{}/plot/classification_map_{}".format(study, label_type), "rb"))
    efficient_labels, efficient_predictions, efficient_scores, label_index_map = np.array(efficient_classification_map["labels"]), np.array(efficient_classification_map["predictions"]), np.array(efficient_classification_map["scores"]), efficient_classification_map["label_index_map"]
    efficient_classification_report = sklearn.metrics.classification_report(efficient_labels, efficient_predictions, digits=3, output_dict=True, zero_division=0)
    for i in range(0, len(efficient_scores)):
        max_i = np.argmax(efficient_scores[i])
        assert(max_i == label_index_map[efficient_predictions[i]])
    ##plot accuracy
    fig, ax = plt.subplots()
    efficient_accuracy = efficient_classification_report.pop("accuracy")
    for item in ["macro avg", "weighted avg", "Empty"]: ##pop Empty as well for proper class count, there aren't any Empty labels but Empty predictions are possible
        efficient_classification_report.pop(item)
    num_classes = len(set(efficient_labels))
    random_accuracy = 1 / float(num_classes)
    xlabels = ["Random Classifier", "MOAProfiler"]
    x = [1,2]
    ax.set_xticks(x)
    accuracies = [random_accuracy, efficient_accuracy]
    bar = ax.bar(x, accuracies, color=("black", "gold"))
    ax.set_xticklabels(xlabels)
    for i,j in zip(x, accuracies):
        ax.annotate("{:.1%}".format(j), xy=(i - .03, j +.03),fontsize=12)
    ax.set_ylabel("Accuracy")
    ax.set_ylim((0,1.03))
    plt.title("{}: Image Field Classification Accuracy over {} Classes".format(study.upper(), num_classes))
    plt.savefig("outputs/classification_accuracy_{}_{}.png".format(study, label_type), dpi=300)
    ##plot micro-averaged PRC
    reverse_map = {value: key for (key, value) in label_index_map.items()} ##index: class
    classes = []
    for i in range(0, len(reverse_map)):
        classes.append(reverse_map[i])
    binary_labels = sklearn.preprocessing.label_binarize(efficient_labels, classes=classes) ##classes arg needs to be the same order as output of the neural network 
    precision, recall, _ = sklearn.metrics.precision_recall_curve(binary_labels.ravel(), efficient_scores.ravel())
    auprc = np.trapz(recall, precision)
    positive_prevalence = sum(binary_labels.ravel()) / float(len(binary_labels.ravel()))
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color="gold", label="Model AUPRC={}".format(round(auprc, 2)))
    ax.axhline(positive_prevalence, ls="--", color="black", label="Random AUPRC={}".format(round(positive_prevalence, 3)))
    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")
    plt.title("{}: Classification PRC over {} Classes".format(study.upper(), num_classes))
    ax.set_xlim((0,1.03))
    ax.set_ylim((0,1.03))
    ax.legend(loc='upper right', prop={"size":9}, bbox_to_anchor=(1, 1.32))
    plt.gcf().subplots_adjust(top=.76)
    plt.savefig("outputs/classification_prc_{}_{}.png".format(study, label_type), dpi=300)

def generateIntraMOAvsInter(study=None, label_type=None, well_aggregator=None, moa_avg=False, metric=None):
    """
    Plots average pairwise similarities of three groups of embedding populations
    """ 
    if label_type == None:
        return
    latent_dictionary = pickle.load(open("pickles/{}/plot/latent_dictionary_label_type_{}_well_aggregated_True_{}_full_dataset.pkl".format(study, label_type, well_aggregator), "rb"))
    latent_dictionary = filterToTestSet(latent_dictionary, csv_map[label_type]["test"], study=study)
    embeddings, labels, _, perturbations = latent_dictionary["embeddings"], latent_dictionary["labels"], latent_dictionary["wells"], latent_dictionary["perturbations"]
    ##drop negatives 
    purge_indices = []
    for i in range(0, len(labels)):
        if labels[i] in ["no_target", "Empty"]: ##JUMP represent with "no_target", lincs uses "Empty"
            purge_indices.append(i)
    embeddings = np.array([embeddings[j] for j in range(0, len(embeddings)) if j not in purge_indices])
    labels = [labels[j] for j in range(0, len(labels)) if j not in purge_indices]
    perturbations = [perturbations[j] for j in range(0, len(perturbations)) if j not in purge_indices]
    print("dropped negative controls, new size: ", len(labels))

    ##logic: get every pair of indices, handle intra-label pairs and inter-label pairs differently 
    same_pert_similarities = []
    same_pert_similarities_moa_specific = {moa: [] for moa in set(labels)}
    intra_similarities = []
    intra_similarities_moa_specific = {moa: [] for moa in set(labels)}
    inter_similarities = []
    indices = [index for index in range(0, len(labels))]
    pairwise_indices = itertools.combinations(indices, 2) ##pairwise (i,j) for all indices
    for index1, index2 in pairwise_indices:
        if index1 == index2:
            continue 
        if metric == "pearson":
            similarity = scipy.stats.pearsonr(embeddings[index1], embeddings[index2])[0]
        if metric == "cosine":
            similarity = 1.0 - scipy.spatial.distance.cosine(embeddings[index1], embeddings[index2])
        if labels[index1] == labels[index2] and perturbations[index1] == perturbations[index2]: #same label, same perturbation
            same_pert_similarities.append(similarity)
            same_pert_similarities_moa_specific[labels[index1]].append(similarity)
        if labels[index1] == labels[index2] and perturbations[index1] != perturbations[index2]: #same label, but different perturbation (intra)
            intra_similarities.append(similarity)
            intra_similarities_moa_specific[labels[index1]].append(similarity)
        if labels[index1] != labels[index2]: ##different label (inter)
            inter_similarities.append(similarity)
            assert(perturbations[index1] != perturbations[index2])
    ##get avg of pairwise similarities for all same-MOA different-perturbation pairs
    if moa_avg: ##if take average within each MOA first, then average the averages
        same_pert_similarities = [np.mean(same_pert_similarities_moa_specific[key]) for key in same_pert_similarities_moa_specific if len(same_pert_similarities_moa_specific[key]) > 0]
        intra_similarities = [np.mean(intra_similarities_moa_specific[key]) for key in intra_similarities_moa_specific if len(intra_similarities_moa_specific[key]) > 0]
    pert_mean, pert_std, pert_size = np.mean(same_pert_similarities), np.std(same_pert_similarities), len(same_pert_similarities)
    intra_size = len(intra_similarities)
    if intra_size != 0: ##compound holdout will not have an intra population because only 1 held-out compound for each MOA (due to limited data size)
        intra_mean, intra_std = np.mean(intra_similarities), np.std(intra_similarities)
    inter_mean, inter_std, inter_size = np.mean(inter_similarities), np.std(inter_similarities), len(inter_similarities)

    ##statistical significance between populations inter and intra
    if intra_size != 0:
        z_val = (intra_mean - inter_mean) / float( np.sqrt( ((intra_std**2) / float(len(intra_similarities))) + ((inter_std**2) / float(len(inter_similarities))))) 
        cdf_one_sided = scipy.stats.norm.cdf(z_val) 
        cdf_two_sided = (scipy.stats.norm.cdf(z_val) * 2) - 1 
        p_val_one_sided = 1 - cdf_one_sided ##we want the one sided with Ho: means are equal, 
        p_val_two_sided = 1 - cdf_two_sided ##we want the one sided with Ho: means are equal, alternative: not equal 
        print("z_val: ", z_val)
        print("p-value: {}, {:E}".format(p_val_two_sided, p_val_two_sided))
    
    fig, ax = plt.subplots()
    if intra_size != 0:
        xlabels = ["Same Compound\nSame MOA", "Different Compound\nSame MOA", "Different Compound\nDifferent MOA"]
        x = np.array([1,2,3])
    else:
        xlabels = ["Same Compound\nSame MOA", "Different Compound\nDifferent MOA"]
        x = np.array([1,2])
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Average Pairwise {} Similarity".format(metric.title()))
    ax.set_ylim((0,1.02))
    width = .3
    if intra_size != 0:
        plt.title("Compound Stratified {} Similarities".format(metric.title()))        
        bar = ax.bar(x, [pert_mean, intra_mean, inter_mean], yerr=[pert_std, intra_std, inter_std], color=("grey", "blue", "orange"))
        for i,j in zip(x, [(pert_mean, pert_size), (intra_mean, intra_size), (inter_mean, inter_size)]):
            if j[0] < 0.01:
                ax.annotate("{:.1E}\nn={:.1E}".format(j[0], j[1]), xy=(i - .41, j[0] +.06),fontsize=8)
            else:
                ax.annotate("{}\nn={:.1E}".format(round(j[0], 2), j[1]), xy=(i - .41, j[0] +.06),fontsize=8)
            x = x + width
        ax.annotate("two sided p ={:.1E}".format(p_val_two_sided), xy=(0,.98), fontsize=8)
    else:
        plt.title("{}: Compound Stratified {} Similarities".format(study.upper(), metric.title()))        
        bar = ax.bar(x, [pert_mean, inter_mean], yerr=[pert_std, inter_std], color=("grey", "orange"))
        for i,j in zip(x, [(pert_mean, pert_size), (inter_mean, inter_size)]):
            if j[0] < 0.01:
                ax.annotate("{:.1E}\nn={:.1E}".format(j[0], j[1]), xy=(i - .41, j[0] +.06),fontsize=8)
            else:
                ax.annotate("{}\nn={:.1E}".format(round(j[0], 2), j[1]), xy=(i - .41, j[0] +.06),fontsize=8)
            x = x + width
    plt.savefig("outputs/intra_vs_inter_{}_{}_{}_{}.png".format(study, label_type, well_aggregator, metric), dpi=300)

def plotKNN(study=None, metric=None, label_type=None, drop_neg_control=None, aggregate_by_well=None, well_aggregator="mean", deep_profile_type=None):
    """
    Plots F1, precision, recall of K-NN 
    """
    efficient_knn = pickle.load(open("pickles/{}/plot/k_pred_map_CP_False_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(study, metric, label_type, drop_neg_control, aggregate_by_well, well_aggregator), "rb"))
    cell_profiler_knn = pickle.load(open("pickles/{}/plot/k_pred_map_CP_True_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}.pkl".format(study, metric, label_type, drop_neg_control, aggregate_by_well), "rb"))
    deep_profiler_knn = pickle.load(open("pickles/{}/plot/k_pred_map_CP_False_DP_True_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(study, metric, label_type, drop_neg_control, aggregate_by_well, deep_profile_type), "rb"))
    min_k = min(len(set(efficient_knn.keys())), len(set(cell_profiler_knn.keys())), len(set(deep_profiler_knn.keys())))
    ks_to_use = list(range(1, min_k))
    efficient_prs = [(k, sklearn.metrics.precision_score(efficient_knn[k]["labels"], efficient_knn[k]["predictions"], average="weighted", zero_division=0),
        sklearn.metrics.recall_score(efficient_knn[k]["labels"], efficient_knn[k]["predictions"], average="weighted", zero_division=0), 
        sklearn.metrics.f1_score(efficient_knn[k]["labels"], efficient_knn[k]["predictions"], average="weighted", zero_division=0)) for k in ks_to_use]
    cp_prs = [(k, sklearn.metrics.precision_score(cell_profiler_knn[k]["labels"], cell_profiler_knn[k]["predictions"], average="weighted", zero_division=0),
        sklearn.metrics.recall_score(cell_profiler_knn[k]["labels"], cell_profiler_knn[k]["predictions"], average="weighted", zero_division=0), 
        sklearn.metrics.f1_score(cell_profiler_knn[k]["labels"], cell_profiler_knn[k]["predictions"], average="weighted", zero_division=0)) for k in ks_to_use]
    dp_prs = [(k, sklearn.metrics.precision_score(deep_profiler_knn[k]["labels"], deep_profiler_knn[k]["predictions"], average="weighted", zero_division=0),
        sklearn.metrics.recall_score(deep_profiler_knn[k]["labels"], deep_profiler_knn[k]["predictions"], average="weighted", zero_division=0),
        sklearn.metrics.f1_score(deep_profiler_knn[k]["labels"], deep_profiler_knn[k]["predictions"], average="weighted", zero_division=0)) for k in ks_to_use]
    maps = {"MOAProfiler": efficient_prs, "CellProfiler": cp_prs, "DeepProfiler": dp_prs}
    color_map = {"CellProfiler":"red", "DeepProfiler":"purple", "MOAProfiler":"gold"}
    ##make separate precion plots and recall plots
    max_values = {plot: {method: 0 for method in sorted(maps.keys())} for plot in ["Precision", "Recall", "F1"]}
    for plot in ["Precision", "Recall", "F1"]:
        # height = .98
        fig, ax = plt.subplots()
        for m in sorted(maps.keys(), reverse=True): ##MP, DP, CP order 
            prs = maps[m]
            ks = [x[0] for x in prs]
            if plot == "Precision":
                y = [x[1] for x in prs]
            if plot == "Recall":
                y = [x[2] for x in prs]
            if plot == "F1":
                y = [x[3] for x in prs]
            max_k = 0 
            for i in range(0, len(ks)):
                if y[i] > max_values[plot][m]:
                    max_values[plot][m] = y[i]
                    max_k = ks[i]
            ax.plot(ks, y, label=m + " ({} @ optimal k={})".format(round(max_values[plot][m], 2), max_k), color=color_map[m])
        print("{} {} {} knn, {} MP improvement over CP: {}, DP: {}".format(study, label_type, plot, [max_values[plot]["CellProfiler"], max_values[plot]["DeepProfiler"], max_values[plot]["MOAProfiler"]] , (max_values[plot]["MOAProfiler"] - max_values[plot]["CellProfiler"]) / max_values[plot]["CellProfiler"], (max_values[plot]["MOAProfiler"] - max_values[plot]["DeepProfiler"]) / max_values[plot]["DeepProfiler"]))
        ax.set_xlabel("k")
        ax.set_ylabel(plot)
        ax.set_ylim((0,1.03))
        ax.legend(loc='upper right', prop={"size":9}, bbox_to_anchor=(1, 1.32))
        plt.title("{}: {}".format(study.upper(), plot))
        plt.gcf().subplots_adjust(top=.76)
        plt.savefig("outputs/{}_{}_{}_{}_drop_neg_{}_well_aggregated_{}_{}_{}.png".format(plot, study, metric, label_type, drop_neg_control, aggregate_by_well, well_aggregator, deep_profile_type), dpi=300)

def plotReduction(deep_profile_type=None, study=None, label_type=None, aggregate_by_well=None, well_aggregator=None, method="PCA", num_components=2):
    """
    Loads latent dictionary, and plots the PCA or TSNE reduction of three example MOAs (chosen because they have 4 or more compounds each): ["CDK inhibitor", "HSP inhibitor", "HDAC inhibitor"]
    """
    for alg in ["CP", "DP", "MP"]:
        if alg == "MP":
            latent_dictionary = pickle.load(open("pickles/{}/plot/latent_dictionary_label_type_{}_well_aggregated_{}_{}_full_dataset.pkl".format(study, label_type, aggregate_by_well, well_aggregator), "rb"))
        if alg == "CP":
            latent_dictionary = pickle.load(open("pickles/{}/plot/CP_latent_dictionary_label_type_{}_full_dataset.pkl".format(study, label_type), "rb"))
        if alg == "DP":
            latent_dictionary = pickle.load(open("pickles/{}/plot/DP_latent_dictionary_label_type_{}_{}_full_dataset.pkl".format(study, label_type, deep_profile_type), "rb"))
            
        latent_dictionary = standardizeEmbeddingsByDMSOPlate(latent_dictionary)
        latent_dictionary = filterToTestSet(latent_dictionary, csv_map[label_type]["test"], study=study)

        embeddings = latent_dictionary["embeddings"]
        labels = latent_dictionary["labels"]
        perturbations = latent_dictionary["perturbations"]
        ##drop negative 
        purge_indices = []
        for i in range(0, len(labels)):
            if labels[i] in ["no_target", "Empty"]: ##JUMP represent with "no_target", lincs uses "Empty"
                purge_indices.append(i)
        embeddings = np.array([embeddings[j] for j in range(0, len(embeddings)) if j not in purge_indices])
        labels = [labels[j] for j in range(0, len(labels)) if j not in purge_indices]
        perturbations = [perturbations[j] for j in range(0, len(perturbations)) if j not in purge_indices]
        ##reduce space to just n classes subset
        n_classes_to_keep = ["CDK inhibitor", "HSP inhibitor", "HDAC inhibitor"]
        indices = [i for i in range(0, len(labels)) if labels[i] in n_classes_to_keep]
        embeddings = np.array([embeddings[i] for i in range(0, len(embeddings)) if i in indices])
        labels = [labels[i] for i in range(0, len(labels)) if i in indices]
        perturbations = [perturbations[i] for i in range(0, len(perturbations)) if i in indices]
        label_set = n_classes_to_keep
        axis_scale = 2
        ##do normalized and un-normalized reduction
        for normalized in [False]:
            for trial in range(0, 3): ##TSNE is stochastic, so let's do this a couple times
                if normalized:
                    X_new = sklearn.preprocessing.normalize(embeddings, axis=0) ##normalize each column (feature)
                else:
                    X_new = embeddings
                if method=="PCA":
                    reducer = sklearn.decomposition.PCA(n_components=num_components, svd_solver="auto") ##set svd_solver to "auto" to speed computation with a randomized Halko et al method, or "full"
                if method == "TSNE":
                    reducer = sklearn.manifold.TSNE(n_components=num_components)
                X_new = reducer.fit_transform(X_new)
                if num_components == 2:
                    df = pd.DataFrame(columns = ["label", "perturbation", "{}1".format(method), "{}2".format(method)])
                    axis1 = X_new[:,0]
                    axis2 = X_new[:,1]
                if num_components == 3:
                    df = pd.DataFrame(columns = ["label", "perturbation", "{}1".format(method), "{}2".format(method), "{}3".format(method)])
                    axis1 = X_new[:,0]
                    axis2 = X_new[:,1]
                    axis3 = X_new[:,2]
                df["label"] = labels
                df["perturbation"] = perturbations
                df["{}1".format(method)] = axis1
                df["{}2".format(method)] = axis2
                if num_components == 3:
                    df["{}3".format(method)] = axis3
                color_maps = {"greens": ["aquamarine", "turquoise", "lightseagreen", "mediumturquoise", "green",  "seagreen", "springgreen", "lime", "forestgreen", "limegreen", "darkgreen"], "oranges": ["blanchedalmond", "papayawhip", "moccasin", "oldlace", "bisque", "darkorange", "burlywood", "tan", "wheat", "navajowhite", "orange"], "blues": ["mediumpurple", "rebeccapurple", "blueviolet", "indigo", "cornflowerblue", "royalblue", "navy", "mediumblue", "darkblue", "blue", "slateblue"]}
                marker_list = ["x", "*", "o"]
                markers = {}
                if num_components == 2:
                    fig, ax = plt.subplots()
                else:
                    fig = plt.figure()
                    ax = plt.axes(projection ="3d")
                ##for each class plot the data            
                min_x, max_x = float("inf"), float("-inf")
                for j in range(0, len(label_set)):
                    scats = []
                    label = label_set[j]
                    if len(color_maps) != 0:
                        color_family = color_maps.pop(sorted(list(color_maps.keys()))[0])
                        markers[label] = marker_list.pop()
                    else: 
                        color_family = ["black", "dimgray", "grey", "darkgrey", "silver", "lightgrey"]
                    df_sub = df[df["label"] == label]
                    perturbations_subset = sorted(list(set(df_sub["perturbation"])), reverse=True) ##sorted list fixes key order
                    for pert in perturbations_subset: ##for each MOA, plot its different perturbations with slightly different colors  
                        df_sub_sub = df_sub[df_sub["perturbation"] == pert]
                        if len(color_family) != 0:
                            color = color_family.pop()
                        else:
                            color = "black" 
                        X = df_sub_sub.drop(["label"], axis=1).drop(["perturbation"], axis=1).to_numpy()
                        axis1 = X[:,0]
                        axis2 = X[:,1]
                        min_x, max_x = min(min(axis1), min_x), max(max(axis1), max_x)
                        if num_components == 3:
                            axis3 = X[:,2]
                        if num_components == 2:
                            scat = ax.scatter(axis1, axis2, color=color, s=9, label=pert + " (" + label + ")", marker=markers[label])
                        else:
                            scat = ax.scatter(axis1, axis2, axis3,color=color, s=9, label=pert, markers=markers[label])
                        scats.append(scat)
                ax.set_xlim((min_x - abs(int(axis_scale * min_x)),max_x))
                ax.set_xlabel("{}1".format(method),  fontsize=9)
                ax.set_ylabel("{}2".format(method), fontsize=9)
                if num_components == 3:
                    ax.set_zlabel("{}3".format(method), fontsize=9)
                ax.legend(loc='upper left', prop={"size":7})
                plt.title("{}: {} Embeddings in {} Space\nColored by Perturbation".format(study.upper(), alg, method), fontsize=12, y=1.02)
                plt.savefig("outputs/reduced_latent_{}_{}_{}_{}_{}_{}_{}_{}_{}.png".format(study, label_type, alg, aggregate_by_well, well_aggregator, method, num_components, normalized, trial), dpi=300)

def plotClassSizeHistogram(study=None, verbose=False):
    """
    Plots MOA-replicate distribution
    """
    if study == "JUMP1":
        df = pd.read_csv("csvs/JUMP1/compound_images_with_MOA_no_polypharm_well_excluded.csv")
    if study == "lincs":
        df = pd.read_csv("csvs/lincs/lincs_ten_micromolar_no_polypharm.csv")
    df = df[df["moas"].ne("Empty")] ##exclude DMSO controls
    plate_well_moa_dict = {} ##key plate_well, value: moa 
    for index, row in df.iterrows():
        plate_well = getBarcode(row["imagename"]) + "_" + getRowColumn(row["imagename"])
        moa = row["moas"]
        if moa in ["Empty", "no_target"]:
            continue
        if plate_well not in plate_well_moa_dict:
            plate_well_moa_dict[plate_well] = moa
        else:
            assert(plate_well_moa_dict[plate_well] == moa)
    moa_set = set(plate_well_moa_dict.values())
    moa_list = list(plate_well_moa_dict.values())
    counts = []
    moa_count_dict = {moa: 0 for moa in moa_set}
    for moa in moa_set:
        c = moa_list.count(moa)
        moa_count_dict[moa] = c
        counts.append(c)
    fig, ax = plt.subplots()
    x = sorted(set(counts))
    y = [counts.count(x_i) for x_i in x]
    sum_y = float(sum(y))
    y = [y_i / sum_y for y_i in y]
    ax.bar(x,y)
    ax.set_xticks(list(range(0, max(x) + 2, 20)))
    ax.set_xlabel("MOA-Replicate Count")
    ax.set_ylabel("Fraction of MOAs")
    plt.title("{}: Distribution of MOA-Replicate Counts".format(study.upper()))
    plt.savefig("outputs/histogram_{}.png".format(study), dpi=300)
    if verbose:
        print(sorted(moa_count_dict.items(), key=lambda x: x[1]))
    
def plotDistributionByAggregatedLatentRep(study=None, class_aggregator="mean", metric=None, label_type=None, drop_neg_control=None, aggregate_by_well=None, well_aggregator="mean", deep_profile_type=None):
    """
    Plots distribution of F1, precision, and recall scores of the class latent assignment metric on embeddings
    """
    cell_profiler_dict = pickle.load(open("pickles/{}/plot/pred_labels_map_{}_latent_CP_True_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}.pkl".format(study, class_aggregator, metric, label_type, drop_neg_control, aggregate_by_well), "rb"))
    deep_profiler_dict = pickle.load(open("pickles/{}/plot/pred_labels_map_{}_latent_CP_False_DP_True_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(study, class_aggregator, metric, label_type, drop_neg_control, aggregate_by_well, deep_profile_type), "rb"))
    efficient_dict = pickle.load(open("pickles/{}/plot/pred_labels_map_{}_latent_CP_False_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(study, class_aggregator, metric, label_type, drop_neg_control, aggregate_by_well, well_aggregator), "rb"))
    methods = ["CellProfiler", "DeepProfiler", "MOAProfiler"]
    score_types = ["precision", "recall", "f1-score"]
    dictionaries = {"CellProfiler": cell_profiler_dict, "DeepProfiler": deep_profiler_dict, "MOAProfiler": efficient_dict}
    colors = {"CellProfiler": "red", "DeepProfiler": "purple", "MOAProfiler": "gold"}
    scores_dict = {method: {score_type: [] for score_type in score_types} for method in methods}
    MP_dict = {score_type: [] for score_type in score_types} #key: score type,value: list of (moa, score) tuples 
    for method in methods:
        dictionary = dictionaries[method]
        classification_report = sklearn.metrics.classification_report(dictionary["labels"], dictionary["predictions"], digits=3, output_dict=True, zero_division=0)
        classification_report.pop("accuracy")
        classification_report.pop("macro avg")
        classification_report.pop("weighted avg")
        x_values = list(classification_report.keys())
        for moa in classification_report:
            for score_type in score_types:
                scores_dict[method][score_type].append(classification_report[moa][score_type])
        if method == "MOAProfiler":
            for moa in classification_report:
                for score_type in score_types: 
                    MP_dict[score_type].append((moa, classification_report[moa][score_type]))  
    for score_type in score_types:
        fig, ax = plt.subplots()
        max_height = 0 
        for method in methods:
            n, bins, patches = ax.hist(scores_dict[method][score_type], color=colors[method], histtype="step", label=method)
            max_height = max(max_height, max(n))
        ##print out top performing MOAs for each score_type 
        mp_sorted = sorted(MP_dict[score_type], key=lambda x: x[1])
        top_scorers = mp_sorted[-5:]
        bottom_scorers = mp_sorted[0:5]
        height = max_height
        for i in range(0, len(top_scorers)):
            top_scorer = top_scorers[i]
            bottom_scorer = bottom_scorers[i]
            if i == 0:
                ax.annotate("bottom MP scorers:", xy=(.20, height),fontsize=6)
                ax.annotate("top MP scorers:", xy=(.63, height),fontsize=6)
                height -= max(int(max_height * .05), .7)
            ax.annotate("{}".format(bottom_scorer[0]), xy=(.20, height),fontsize=6)
            ax.annotate("{}".format(top_scorer[0]), xy=(.63, height),fontsize=6)
            height -= max(int(max_height * .05), .7)

        ax.set_ylabel("Number of MOAs")
        ax.set_xlabel(score_type.capitalize())
        plt.title("{}: Distribution of {} Scores for {} MOAs".format(study.upper(), score_type.capitalize(), len(classification_report)))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
        ax.legend(loc='upper right', prop={"size":10}, bbox_to_anchor=(1, 1.32))
        plt.gcf().subplots_adjust(top=.76)
        plt.savefig("outputs/histogram_clustering_{}_{}_{}_{}_{}_{}_{}_{}_{}.png".format(score_type.capitalize(), study, class_aggregator, metric, label_type, drop_neg_control, aggregate_by_well, well_aggregator, deep_profile_type), dpi=300)

def plotEnrichment(study=None, class_aggregator="mean", metric=None, label_type=None, drop_neg_control=None, aggregate_by_well=None, well_aggregator="mean", deep_profile_type=None):
    """
    Plots enrichment for CP vs DP vs MP
    """
    cell_profiler_enrichment = pickle.load(open("pickles/{}/plot/enrichment_{}_latent_CP_True_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}.pkl".format(study, class_aggregator, metric, label_type, drop_neg_control, aggregate_by_well), "rb"))
    deep_profiler_enrichment = pickle.load(open("pickles/{}/plot/enrichment_{}_latent_CP_False_DP_True_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(study, class_aggregator, metric, label_type, drop_neg_control, aggregate_by_well, deep_profile_type), "rb"))
    efficient_enrichment = pickle.load(open("pickles/{}/plot/enrichment_{}_latent_CP_False_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(study, class_aggregator, metric, label_type, drop_neg_control, aggregate_by_well, well_aggregator), "rb"))
    ##keys are not perfectly at .2 increments, i.e. 99.00000000000001
    cell_profiler_enrichment = {round(key, 2): cell_profiler_enrichment[key] for key in cell_profiler_enrichment}
    deep_profiler_enrichment = {round(key, 2): deep_profiler_enrichment[key] for key in deep_profiler_enrichment}
    efficient_enrichment = {round(key, 2): efficient_enrichment[key] for key in efficient_enrichment}
    color_map = {"CellProfiler":"red", "DeepProfiler":"purple", "MOAProfiler":"gold"}
    mapp = {"CellProfiler":cell_profiler_enrichment, "DeepProfiler":deep_profiler_enrichment, "MOAProfiler":efficient_enrichment}
    fig, ax = plt.subplots()   
    for m in sorted(mapp.keys(), reverse=True):
        enrichment_dict = mapp[m]
        x = []
        ys = []
        for perc in sorted(list(enrichment_dict.keys())):
            x.append(perc)
            ys.append(enrichment_dict[perc])
        x = np.array(x)
        first_percentile = enrichment_dict[99.0]
        ax.plot(x, ys, label=m + " ({} @99th percentile)".format(round(first_percentile, 1)), color=color_map[m]) 
    ax.set_xticks(x)
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Enrichment")
    plt.title("{}: Enrichment Comparison".format(study.upper()))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', prop={"size":10}, bbox_to_anchor=(1, 1.32))
    plt.gcf().subplots_adjust(top=.76)
    plt.savefig("outputs/enrichment_{}_latent_rep_{}_{}_{}_drop_neg_{}_well_aggregated_{}_{}_{}.png".format(class_aggregator, study, metric, label_type, drop_neg_control, aggregate_by_well, well_aggregator, deep_profile_type), dpi=300)

def plotMOATestSpread(study="JUMP1"):
    """
    Plots distribution of plates that had test wells on them 
    """
    if study == "JUMP1":
        df = pd.read_csv("csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test_wells_excluded.csv")
    if study == "lincs":
        df = pd.read_csv("csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_test.csv")
    df = df[df["moas"].ne("Empty")] ##exclude DMSO controls
    moas = set(df["moas"])
    moa_test_map_plates = {moa: set() for moa in moas} ##moa: set of test plate barcodes 
    for index, row in df.iterrows():
        imagename = row["imagename"]
        moa_test_map_plates[row["moas"]].add(getBarcode(imagename))
    counts = [len(moa_test_map_plates[moa]) for moa in moa_test_map_plates] ##for each moa, count the # of plates used for test set
    ##make histogram 
    fig, ax = plt.subplots()  
    ##make normal histogram
    x = sorted(set(counts))
    y = [counts.count(x_i) for x_i in x]
    sum_y = float(sum(y))
    y = [y_i / sum_y for y_i in y] ##normalize counts to frequencies 
    ax.bar(x, y, color="green", label="non-cumulative")
    ##make cumulative histogram
    y_cum = [y[0]]
    summation = y[0]
    for i in range(1, len(y)):
        y_cum.append(y[i] + summation)
        summation = summation + y[i]
    ax.scatter(x, y_cum, edgecolor="purple", color="None", label="cumulative", s=5)
    ax.set_ylabel("Fraction of MOAs")
    ax.set_xlabel("Number of Plates")
    plt.title("{}: Distribution of Test Plates\nfor each MOA".format(study.upper()))
    plt.savefig("outputs/histogram_test_plate_{}.png".format(study.upper()), dpi=300)

def plotMOAPertSpread(study="JUMP1"):
    """
    Plots distribution of compounds for each MOA
    """
    if study == "JUMP1":
        df = pd.read_csv("csvs/JUMP1/compound_images_with_MOA_no_polypharm_well_excluded.csv")
    if study == "lincs":
        df = pd.read_csv("csvs/lincs/lincs_ten_micromolar_no_polypharm.csv")
    df = df[df["moas"].ne("Empty")] ##exclude DMSO controls
    moa_to_pert = {moa: set() for moa in set(df["moas"])} ##key: moa, value: set of perturbations 
    for index, row in df.iterrows():
        moa_to_pert[row["moas"]].add(row["perturbation"])
    counts = [len(moa_to_pert[moa]) for moa in moa_to_pert] ##for each moa, count the # of perturbations
    ##make histogram 
    fig, ax = plt.subplots()   
    x = sorted(set(counts))
    y = [counts.count(x_i) for x_i in x]
    ax.bar(x, y, color="purple")
    ax.set_xticks(list(range(min(counts), max(counts) + 1, 2)))
    ax.set_ylabel("Number of MOAs")
    ax.set_xlabel("Number of Compounds")
    plt.title("{}: Distribution of Compounds\nfor each MOA".format(study.upper()))
    plt.savefig("outputs/histogram_perturbation_{}.png".format(study.upper()), dpi=300)
    
def generateIntraPlatevsInter(study=None, label_type=None, deep_profile_type=None, well_aggregator=None, metric=None):
    """
    Plots 3 plots according to these 3 schemes of same plate vs different plate:
    1) same plate vs different plate (take all the well pairs regardless of MOA)
    2) same plate, same MOA vs different plate, same MOA (take all the well pairs with the same MOA)
    3) same plate, different MOA vs different plate, different MOA (take all the well pairs with different MOA)
    """ 
    if label_type == None:
        return
    latent_types = ["CellProfiler", "DeepProfiler", "MOAProfiler"]
    color_map = {"CellProfiler":"red", "DeepProfiler":"purple", "MOAProfiler":"gold"}
    plot_map = {ps: {lt: {"inter" : -1, "intra": -1} for lt in latent_types} for ps in [1,2,3]} ##key: plate_scheme (1,2,3) key: latent_type, key: "inter" or "intra", value: pairwise similarity avg, std 
    for latent_type in latent_types:
        if latent_type == "MOAProfiler":
            latent_dictionary = pickle.load(open("pickles/{}/plot/latent_dictionary_label_type_{}_well_aggregated_True_{}_full_dataset.pkl".format(study, label_type, well_aggregator), "rb"))
        if latent_type == "CellProfiler":
            latent_dictionary = pickle.load(open("pickles/{}/plot/CP_latent_dictionary_label_type_{}_full_dataset.pkl".format(study, label_type), "rb"))
        if latent_type == "DeepProfiler":
            latent_dictionary = pickle.load(open("pickles/{}/plot/DP_latent_dictionary_label_type_{}_{}_full_dataset.pkl".format(study, label_type, deep_profile_type), "rb"))
        latent_dictionary = standardizeEmbeddingsByDMSOPlate(latent_dictionary)
        latent_dictionary = removeSingleCompoundMOAEmbeddings(latent_dictionary)
        latent_dictionary = filterToTestSet(latent_dictionary, csv_map[label_type]["test"], study=study)
        embeddings, labels, wells, perturbations = latent_dictionary["embeddings"], latent_dictionary["labels"], latent_dictionary["wells"], latent_dictionary["perturbations"]
        ##drop negatives 
        purge_indices = []
        for i in range(0, len(labels)):
            if labels[i] in ["no_target", "Empty"]: ##JUMP represent with "no_target", lincs uses "Empty"
                purge_indices.append(i)
        embeddings = np.array([embeddings[j] for j in range(0, len(embeddings)) if j not in purge_indices])
        labels = [labels[j] for j in range(0, len(labels)) if j not in purge_indices]
        perturbations = [perturbations[j] for j in range(0, len(perturbations)) if j not in purge_indices]
        wells = [wells[j] for j in range(0, len(wells)) if j not in purge_indices]
        plates = [getBarcode(well) for well in wells]
        indices = [index for index in range(0, len(labels))]
        ##logic: get every pair of indices, handle intra-label pairs and inter-label pairs differently 
        for plate_scheme in [1,2,3]:
            intra_similarities = []
            inter_similarities = []
            pairwise_indices = itertools.combinations(indices, 2) ##pairwise (i,j) for all indices
            ##non-parallel version 
            for index1, index2 in pairwise_indices:
                if index1 == index2:
                    continue 
                if metric == "pearson":
                    sim_score = scipy.stats.pearsonr(embeddings[index1], embeddings[index2])[0]
                if metric == "cosine":
                    sim_score = 1.0 - scipy.spatial.distance.cosine(embeddings[index1], embeddings[index2])[0]
                if plate_scheme == 1: 
                    if plates[index1] == plates[index2]: ##same plate
                        intra_similarities.append(sim_score)
                    if plates[index1] != plates[index2]: ##different plate
                        inter_similarities.append(sim_score)
                if plate_scheme == 2:
                    if plates[index1] == plates[index2] and labels[index1] == labels[index2]: ##same plate, same moa  
                        intra_similarities.append(sim_score)
                    if plates[index1] != plates[index2] and labels[index1] == labels[index2]: ##different plate, same moa
                        inter_similarities.append(sim_score)
                if plate_scheme == 3:
                    if plates[index1] == plates[index2] and labels[index1] != labels[index2]: ##same plate, different moa  
                        intra_similarities.append(sim_score)
                    if plates[index1] != plates[index2] and labels[index1] != labels[index2]: ##different plate, different moa
                        inter_similarities.append(sim_score)
            ##get avg of pairwise similarities for all same-MOA different-perturbation pairs
            intra_mean, intra_std = np.mean(intra_similarities), np.std(intra_similarities)
            inter_mean, inter_std = np.mean(inter_similarities), np.std(inter_similarities)
            ##statistical significance between populations inter and intra
            z_val = (intra_mean - inter_mean) / float( np.sqrt( ((intra_std**2) / float(len(intra_similarities))) + ((inter_std**2) / float(len(inter_similarities))))) 
            cdf_one_sided = scipy.stats.norm.cdf(z_val) 
            cdf_two_sided = (scipy.stats.norm.cdf(z_val) * 2) - 1 
            p_val_one_sided = 1 - cdf_one_sided ##we want the one sided with Ho: means are equal, 
            p_val_two_sided = 1 - cdf_two_sided ##we want the one sided with Ho: means are equal, alternative: not equal 
            plot_map[plate_scheme][latent_type]["inter"] = (inter_mean, inter_std)  
            plot_map[plate_scheme][latent_type]["intra"] = (intra_mean, intra_std)   
    ##make plots
    for plate_scheme in [1,2,3]:
        fig, ax = plt.subplots()
        if plate_scheme == 1:
            xlabels = ["Same Plate", "Different Plate"]
        if plate_scheme == 2:
            xlabels = ["Same Plate\nSame MOA", "Different Plate\nSame MOA"]
        if plate_scheme == 3:
            xlabels = ["Same Plate\nDifferent MOA", "Different Plate\nDifferent MOA"]
        x = np.array([1,2])
        ax.set_xticks(x)
        for latent_type in latent_types:
            means = [] ##intra, inter
            stds = []
            means.append(plot_map[plate_scheme][latent_type]["intra"][0])
            means.append(plot_map[plate_scheme][latent_type]["inter"][0])
            stds.append(plot_map[plate_scheme][latent_type]["intra"][1])
            stds.append(plot_map[plate_scheme][latent_type]["inter"][1])
            bar = ax.bar(x, means, yerr=stds, width = .15, color=color_map[latent_type], label=latent_type)
            x = x + .15
            #annotate
            for i,j in zip(x, means):
                if j < 0.01:
                    ax.annotate("{:.1E}".format(j), xy=(i - .20, j +.03),fontsize=8)
                else:
                    ax.annotate("{:.0%}".format(j), xy=(i - .20, j +.03),fontsize=8)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylabel("Average Pairwise {} Similarity".format(metric.title()))
        ax.set_ylim((0,1.02))
        plt.title("{}: Intra-Plate vs Inter-Plate {} Similarity".format(study, metric.title()))        
        plt.savefig("outputs/plate_intra_vs_inter_{}_{}_{}_{}_{}_{}_{}.png".format(study, latent_type, label_type, deep_profile_type, well_aggregator, plate_scheme, metric), dpi=300)
    ##just MOAProfiler plots 
    plate_scheme_color_map = {1: "grey", 2: "orange", 3:"limegreen"} 
    plate_scheme_label_map = {1: "All Embeddings", 2:"Same MOA", 3:"Different MOA"}
    fig, ax = plt.subplots()
    x = np.array([1,2])
    ax.set_xticks(x)
    for plate_scheme in [1,2,3]:
        y = [plot_map[plate_scheme]["MOAProfiler"]["intra"][0], plot_map[plate_scheme]["MOAProfiler"]["inter"][0]]
        stds = [plot_map[plate_scheme]["MOAProfiler"]["intra"][1], plot_map[plate_scheme]["MOAProfiler"]["inter"][1]]
        ax.bar(x, y, yerr=stds, width=.15, color=plate_scheme_color_map[plate_scheme], label=plate_scheme_label_map[plate_scheme])
        #annotate
        for i,j in zip(x, y):
            if j < 0.01:
                ax.annotate("{:.1E}".format(j), xy=(i - .07, j +.03),fontsize=8)
            else:
                ax.annotate("{:.0%}".format(j), xy=(i - .07, j +.03),fontsize=8)
        x = x + .15
    ax.set_xticklabels(["Same Plate", "Different Plate"], fontsize=8)
    ax.set_ylabel("Average Pairwise {} Similarity".format(metric.title()))
    plt.title("{}: Intra-Plate vs Inter-Plate\n{} Similarity".format(study.upper(), metric.title()))        
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', prop={"size":7}, bbox_to_anchor=(1, 1.32))
    plt.gcf().subplots_adjust(top=.76)
    plt.savefig("outputs/plate_intra_vs_inter_MOAProfiler_{}_{}_{}_{}_{}_{}.png".format(study, latent_type, label_type, well_aggregator, plate_scheme, metric), dpi=300)

def plotReducedColoredByPlate(study=None, label_type=None, aggregate_by_well=None, well_aggregator=None, method="PCA", num_components=2, n_plates=2):
    """
    Plots visualization of embeddings colored by plate
    """
    latent_dictionary = pickle.load(open("pickles/{}/plot/latent_dictionary_label_type_{}_well_aggregated_{}_{}_full_dataset.pkl".format(study, label_type, aggregate_by_well, well_aggregator), "rb"))

    latent_dictionary = standardizeEmbeddingsByDMSOPlate(latent_dictionary)
    latent_dictionary = filterToTestSet(latent_dictionary, csv_map[label_type]["test"], study=study)
    latent_dictionary = removeSingleCompoundMOAEmbeddings(latent_dictionary)

    embeddings = latent_dictionary["embeddings"]
    labels = latent_dictionary["labels"]
    wells = latent_dictionary["wells"]
    ##drop negative 
    purge_indices = []
    for i in range(0, len(labels)):
        if labels[i] in ["no_target", "Empty"]: ##JUMP represent with "no_target", lincs uses "Empty"
            purge_indices.append(i)
    embeddings = np.array([embeddings[j] for j in range(0, len(embeddings)) if j not in purge_indices])
    wells = [wells[j] for j in range(0, len(wells)) if j not in purge_indices]
    plates = [getBarcode(well) for well in wells]
    ##reduce space to just n plates
    shuffled_plates = plates.copy()
    random.shuffle(shuffled_plates)
    subset_plates = shuffled_plates[0:n_plates]
    purge_indices = []
    for i in range(0, len(plates)):
        if plates[i] not in subset_plates: ##JUMP represent with "no_target", lincs uses "Empty"
            purge_indices.append(i)
    embeddings = np.array([embeddings[j] for j in range(0, len(embeddings)) if j not in purge_indices])
    plates = [plates[j] for j in range(0, len(plates)) if j not in purge_indices]
    ##plot
    for normalized in [False]:
        for trial in range(0, 5): ##TSNE is stochastic, so let's do this a couple times
            colors=["black", "yellow", "brown", "pink", "grey", "purple", "orange", "red", "blue"]
            # random.shuffle(colors)
            if normalized:
                X_new = sklearn.preprocessing.normalize(embeddings, axis=0) ##normalize each column (feature)
            else:
                X_new = embeddings
            if X_new.shape[0] == 0:
                continue
            print("ZZZ", X_new.shape)
            if method=="PCA":
                reducer = sklearn.decomposition.PCA(n_components=num_components, svd_solver="auto") ##set svd_solver to "auto" to speed computation with a randomized Halko et al method, or "full"
            if method == "TSNE":
                reducer = sklearn.manifold.TSNE(n_components=num_components)
            X_new = reducer.fit_transform(X_new)
            if num_components == 2:
                df = pd.DataFrame(columns = ["plate", "{}1".format(method), "{}2".format(method)])
                axis1 = X_new[:,0]
                axis2 = X_new[:,1]
            if num_components == 3:
                df = pd.DataFrame(columns = ["plate", "{}1".format(method), "{}2".format(method), "{}3".format(method)])
                axis1 = X_new[:,0]
                axis2 = X_new[:,1]
                axis3 = X_new[:,2]
            df["plate"] = plates
            df["{}1".format(method)] = axis1
            df["{}2".format(method)] = axis2
            if num_components == 3:
                df["{}3".format(method)] = axis3
            if num_components == 2:
                fig, ax = plt.subplots()
            else:
                fig = plt.figure()
                ax = plt.axes(projection ="3d")
            ##for each class plot the data            
            min_x, max_x = float("inf"), float("-inf")
            for j in range(0, len(subset_plates)):
                color = colors.pop()
                scats = []
                label = subset_plates[j]
                df_sub = df[df["plate"] == label]
                X = df_sub.drop(["plate"], axis=1).to_numpy()
                axis1 = X[:,0]
                axis2 = X[:,1]
                min_x, max_x = min(min(axis1), min_x), max(max(axis1), max_x)
                if num_components == 3:
                    axis3 = X[:,2]
                if num_components == 2:
                    scat = ax.scatter(axis1, axis2, color=color, s=9, label=label)
                else:
                    scat = ax.scatter(axis1, axis2, axis3,color=color, s=9, label=label)
                scats.append(scat)
            ax.set_xlim((min_x - abs(int(1 * min_x)),max_x))
            ax.set_xlabel("{}1".format(method),  fontsize=9)
            ax.set_ylabel("{}2".format(method), fontsize=9)
            if num_components == 3:
                ax.set_zlabel("{}3".format(method), fontsize=9)
            ax.legend(loc='upper left', prop={"size":6})
            plt.title("Embeddings in {} Space\nColored by Plate".format(method), fontsize=12, y=1.02)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
            ax.legend(loc='upper right', prop={"size":7}, bbox_to_anchor=(1, 1.32))
            plt.gcf().subplots_adjust(top=.76)
            plt.savefig("outputs/reduced_latent_by_plate_{}_{}_{}_{}_{}_{}_{}_{}.png".format(study, label_type, aggregate_by_well, well_aggregator, method, num_components, normalized, trial), dpi=300)

def plotReplicateTrainingPerformances(study="JUMP1", label_type=None):
    """
    Plots performance of smaller training sets experiment
    """
    efficient_classification_map = pickle.load(open("pickles/{}/plot/classification_map_{}".format(study, label_type), "rb"))
    efficient_labels, efficient_predictions, efficient_scores, label_index_map = np.array(efficient_classification_map["labels"]), np.array(efficient_classification_map["predictions"]), np.array(efficient_classification_map["scores"]), efficient_classification_map["label_index_map"]
    efficient_classification_report = sklearn.metrics.classification_report(efficient_labels, efficient_predictions, digits=3, output_dict=True, zero_division=0)
    efficient_accuracy = efficient_classification_report.pop("accuracy")
    if study == "JUMP1":
        x = list(range(4, 36, 4))
        y = [.269, .395, .459, .483, .485, .515, .49, .487]
        full_training_image_length = len(set(pd.read_csv("csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_train.csv")["imagename"]))
        training_image_lengths = [len(set(pd.read_csv(df_i)["imagename"])) for df_i in ["csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_replicates={}_train.csv".format(x_i) for x_i in x ]]
    if study == "lincs":
        x = list(range(1, 8))
        y = [0.234, 0.362, 0.422, 0.428, 0.432, 0.432, 0.43]
        full_training_image_length = len(set(pd.read_csv("csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_train.csv")["imagename"]))
        training_image_lengths = [len(set(pd.read_csv(df_i)["imagename"])) for df_i in ["csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_replicates={}_train.csv".format(x_i) for x_i in x ]]
    training_image_lengths = [t_i / float(full_training_image_length) for t_i in training_image_lengths]
    fig, ax = plt.subplots()
    ax.plot(x, y, marker="x", label="Accuracy over Test Set")
    ax.scatter(x, training_image_lengths, marker="x", color="orange", label="Percentage of Training Set")
    ax.hlines(efficient_accuracy, min(x), max(x), linestyles="--", color="grey")
    plt.title("{}: Performance by Number of Compound Replicates\nin Training Set".format(study.upper()))
    ax.set_xlabel("Maximal Allotted Compound Replicates in Training Set")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', prop={"size":8}, bbox_to_anchor=(1, 1.32))
    plt.gcf().subplots_adjust(top=.76)
    if study == "JUMP1":
        plt.xticks(np.arange(min(x), max(x)+1, 4.0))
    plt.savefig("outputs/{}_MOA_replicate_performance.png".format(study), dpi=300)

def plotCompoundReplicateDistribution(study="JUMP1"):
    """
    Plots distribution of compound replicates
    """
    if study == "JUMP1":
        csv_name = "csvs/JUMP1/compound_images_with_MOA_no_polypharm_well_excluded.csv"
    if study == "lincs":
        csv_name = "csvs/lincs/lincs_ten_micromolar_no_polypharm.csv"
    df = pd.read_csv(csv_name)
    df = df[df["moas"].ne("Empty")] ##exclude DMSO controls
    perturbation_to_wells = {}
    for index, row in df.iterrows():
        perturbation = row["perturbation"]
        plate_well = getBarcode(row["imagename"]) + getRowColumn(row["imagename"])
        if perturbation not in perturbation_to_wells:
            perturbation_to_wells[perturbation] = set([plate_well])
        else:
            perturbation_to_wells[perturbation].add(plate_well)
    perturbation_to_count = {key: len(perturbation_to_wells[key]) for key in perturbation_to_wells}
    counts = list(perturbation_to_count.values())
    fig, ax = plt.subplots()
    x = sorted(set(counts))
    y = [counts.count(x_i) for x_i  in x]
    ax.bar(x, y)
    print("compound replicate {}".format(study), x, y)
    ax.set_xlabel("Number of Replicate Wells")
    ax.set_ylabel("Number of Compounds")
    plt.title("{}: Distribution of Compound Replicates".format(study.upper()))
    plt.savefig("outputs/compound_replicates_{}.png".format(study), dpi=300)

def plotFieldsPerWellDistribution(study="JUMP1"):
    """
    Plots distribution of images per well
    """
    if study == "JUMP1":
        csv_name = "csvs/JUMP1/compound_images_with_MOA_no_polypharm_well_excluded.csv"
    if study == "lincs":
        csv_name = "csvs/lincs/lincs_ten_micromolar_no_polypharm.csv"
    df = pd.read_csv(csv_name)
    df = df[df["moas"].ne("Empty")] ##exclude DMSO controls
    plate_well_to_field = {}
    for index, row in df.iterrows():
        plate_well = getBarcode(row["imagename"]) + getRowColumn(row["imagename"])
        field = getField(row["imagename"])
        if plate_well not in plate_well_to_field:
            plate_well_to_field[plate_well] = set([field])
        else:
            plate_well_to_field[plate_well].add(field)
    plate_well_to_field = {key: len(plate_well_to_field[key]) for key in plate_well_to_field}
    values = list(plate_well_to_field.values())
    x = sorted(set(values))
    y = [values.count(x_i) for x_i in x]
    fig, ax = plt.subplots()
    ax.bar(x, y)
    print("fields per well {}".format(study), x, y)
    ax.set_xlabel("Number of Images Present")
    ax.set_ylabel("Number of Wells")
    plt.title("{}: Distribution of Images per Well".format(study.upper()))
    plt.savefig("outputs/images_per_well_{}.png".format(study), dpi=300)

def plotCompoundHoldoutKPrediction(study="JUMP1", label_type=None):
    """
    Plots results when predicting compound holdout by direct model well classification from image data
    """
    k_map = pickle.load(open("pickles/{}/plot/compound_holdout_model_pred_well_k_map_{}.pkl".format(study, label_type), "rb"))
    if study == "JUMP1":
        class_size = 176
        num_compounds = 59
    if study == "lincs":
        class_size = 424
        num_compounds = 215
    x, y = [], []
    y_random = []
    for k in range(1, 6):# sorted(k_map):
        if k == 1:
            print("{} k=1 accuracy: {}".format(study, k_map[1][0]))
        x.append(k)
        y.append(k_map[k][0])
        y_random.append(k / float(class_size))
    print("{} model classification top-k accuracies {}".format(study, y))
    fig, ax = plt.subplots()
    ax.plot(x, y, "-*", label="MOAProfiler", color="gold")
    ax.plot(x,y_random, "--", label="Random", color="grey")
    plt.title("{}: Held-out Compound Accuracy \nBy Model Classification Output (n={} Compounds)".format(study.upper(), num_compounds))
    ax.set_xlabel("k")
    ax.set_xticks(x)
    ax.set_ylabel("Accuracy")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', prop={"size":8}, bbox_to_anchor=(1, 1.32))
    plt.gcf().subplots_adjust(top=.76)
    plt.savefig("outputs/{}_{}_compound_holdout_k_pred.png".format(study, label_type), dpi=300)
       
def plotCompoundHoldoutKPredictionByLatentBars(study="JUMP1", label_type=None, deep_profile_type=None, class_aggregator=None, metric=None):
    """
    Plots results when predicting compound holdout by model latent representations
    """
    if study == "JUMP1":
        num_compounds = 59
    else:
        num_compounds = 215
    width = 0.18
    score_index_map = {0: "F1", 1:"Precision", 2:"Recall", 3:"Accuracy"}
    color_map = {0: "grey", 1:"salmon", 2:"lightsteelblue", 3:"orange"}
    for prediction_method in ["individual vote", "aggregated vote"]:
        fig, ax = plt.subplots()
        x = np.array([1,2,3])
        ax.set_xticks(x)

        if prediction_method == "individual vote":
            k_map = pickle.load(open("pickles/{}/plot/latent_vote_by_wells_k_map_{}_False_False_{}_{}_True_True_None.pkl".format(study, class_aggregator, metric, label_type), "rb"))
            cp_k_map = pickle.load(open("pickles/{}/plot/latent_vote_by_wells_k_map_{}_True_False_{}_{}_True_True_None.pkl".format(study, class_aggregator, metric, label_type), "rb"))
            dp_k_map = pickle.load(open("pickles/{}/plot/latent_vote_by_wells_k_map_{}_False_True_{}_{}_True_True_{}.pkl".format(study, class_aggregator, metric, label_type, deep_profile_type), "rb"))
        if prediction_method == "aggregated vote":
            k_map = pickle.load(open("pickles/{}/plot/latent_vote_by_embedding_{}_False_False_{}_{}_True_True_None.pkl".format(study, class_aggregator, metric, label_type), "rb"))
            cp_k_map = pickle.load(open("pickles/{}/plot/latent_vote_by_embedding_{}_True_False_{}_{}_True_True_None.pkl".format(study, class_aggregator, metric, label_type), "rb"))
            dp_k_map = pickle.load(open("pickles/{}/plot/latent_vote_by_embedding_{}_False_True_{}_{}_True_True_{}.pkl".format(study, class_aggregator, metric, label_type, deep_profile_type), "rb"))

        for score_index in [0,1,2,3]: ##don't plot accuracy, same as recall in this case
            mp_score = k_map[1][0][score_index]
            dp_score = dp_k_map[1][0][score_index]
            cp_score = cp_k_map[1][0][score_index]
            y = [cp_score, dp_score, mp_score]
            ax.bar(x, y, width=width, color=color_map[score_index], label=score_index_map[score_index])
            ##annotate values
            for i,j in zip(x, y):
                ax.annotate("{:.1%}".format(j), xy=(i - .06, j +.001),fontsize=6.5)
            x = x + width 
        xlabels = ["CellProfiler", "DeepProfiler", "MOAProfiler"]
        ax.set_xticklabels(xlabels)
        plt.title("{}: Held-out Compound Accuracy\nBy Latent Similarity, {} (n={} Compounds)".format(study.upper(), prediction_method.title(), num_compounds))
        ax.set_ylabel("Score")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
        ax.legend(loc='upper right', prop={"size":7.5}, bbox_to_anchor=(1, 1.35))
        plt.gcf().subplots_adjust(top=.76)
        plt.savefig("outputs/{}_{}_compound_holdout_latent_k_pred_bar_combined_{}_{}_{}.png".format(study, label_type, deep_profile_type, metric, prediction_method), dpi=300)

def plotCompoundHoldoutMOABreakdown(study="JUMP1", label_type=None, deep_profile_type=None, class_aggregator=None, metric=None):
    k_map = pickle.load(open("pickles/{}/plot/latent_vote_by_embedding_{}_False_False_{}_{}_True_True_None.pkl".format(study, class_aggregator, metric, label_type), "rb"))
    cp_k_map = pickle.load(open("pickles/{}/plot/latent_vote_by_embedding_{}_True_False_{}_{}_True_True_None.pkl".format(study, class_aggregator, metric, label_type), "rb"))
    dp_k_map = pickle.load(open("pickles/{}/plot/latent_vote_by_embedding_{}_False_True_{}_{}_True_True_{}.pkl".format(study, class_aggregator, metric, label_type, deep_profile_type), "rb"))
    mapp = {"CP": cp_k_map, "DP": dp_k_map, "MP": k_map}
    moa_to_correct = {moa: [0, 0, 0] for moa in set(k_map[1][1][1])} ##key: moa, value: [is_correct CP, is_correct DP, is_correct MP]
    for index, method in enumerate(["CP", "DP", "MP"]):
        km = mapp[method]
        predictions = km[1][1][0]
        labels = km[1][1][1]
        for j in range(0, len(labels)):
            if labels[j] == predictions[j]:
                moa_to_correct[labels[j]][index] += 1
    # print(moa_to_correct)
    ##make dataframe and write to CSV 
    moas = sorted(list(set(moa_to_correct.keys())))
    CPs = [moa_to_correct[moa][0] for moa in moas]
    DPs = [moa_to_correct[moa][1] for moa in moas]
    MPs = [moa_to_correct[moa][2] for moa in moas]
    df = pd.DataFrame(list(zip(moas, CPs, DPs, MPs)), columns=["moa", "CP", "DP", "MP"])
    # print(df)
    # print("CP: ", sum(list(df["CP"])) / float(len(df)))
    # print("DP: ", sum(list(df["DP"])) / float(len(df)))
    # print("MP: ", sum(list(df["MP"])) / float(len(df)))
    ##add column for # of training compounds for each moa, see if there is a correlation between sample size and performance 
    if study == "JUMP1":
        train_df = pd.read_csv("csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_well_excluded_compound_holdout_train_2_balanced_moas.csv")
    if study == "lincs":
        train_df = pd.read_csv("csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_compound_holdout_train_2_balanced_moas.csv")
    moa_to_compounds = {m:set() for m in set(train_df["moas"])}
    for index, row in train_df.iterrows():
        moa_to_compounds[row["moas"]].add(row["perturbation"])
    compound_cardinalities = [len(moa_to_compounds[moa]) for moa in list(df["moa"])]
    df["# of training compounds"] = compound_cardinalities
    df = df.sort_values("# of training compounds")
    df.to_csv("outputs/compound_holdout_moa_breakdonwn_{}_{}_{}.csv".format(study, label_type, deep_profile_type), index=False)
    ##break down MP accuracies by # of training compounds
    x = []
    y = []
    ns = []
    # for cardinality in set(df["# of training compounds"]):
    for cardinality in range(1, max(df["# of training compounds"]) + 1):
        sub_df = df[df["# of training compounds"] == cardinality]
        if len(sub_df) > 0:
            sub_df_accuracy = sum(list(sub_df["MP"])) / float(len(sub_df))
        else:
            sub_df_accuracy = 0
        x.append(cardinality)
        y.append(sub_df_accuracy)
        ns.append(len(sub_df))
    fig, ax = plt.subplots()
    ax.bar(x, y)
    if study == "JUMP1":
        ax.set_ylim((0, max(y) + .05))
    else:
        ax.set_ylim((0, max(y) + .08))
    for i,j,k in zip(x, y, ns):
        if study == "JUMP1":
            ax.annotate("n={}".format(k), xy=(i - .10 , j + .005),fontsize=8)
        else:
            ax.annotate("n={}".format(k), xy=(i - .50, j + .006),fontsize=8, rotation=270)
    ax.set_xlabel("Number of Training Compounds")
    ax.set_ylabel("Fraction of Compounds Correctly Predicted")
    plt.title("{}: Accuracy by Number of Training Compounds".format(study.upper()))
    plt.savefig("outputs/compound_holdout_moa_breakdonwn_{}_{}_{}.png".format(study, label_type, deep_profile_type))

def plotLogisticRegression(study=None, class_aggregator="median", metric=None, label_type=None, drop_neg_control=None, aggregate_by_well=None, well_aggregator="mean", deep_profile_type=None):
    CP = pickle.load(open("pickles/{}/plot/logistic_regression_{}_CP_True_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}.pkl".format(study, class_aggregator, metric, label_type, drop_neg_control, aggregate_by_well), "rb"))
    DP = pickle.load(open("pickles/{}/plot/logistic_regression_{}_CP_False_DP_True_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(study, class_aggregator, metric, label_type, drop_neg_control, aggregate_by_well, deep_profile_type), "rb"))
    MP = pickle.load(open("pickles/{}/plot/logistic_regression_{}_CP_False_DP_False_{}_{}_drop_neg_{}_well_aggregated_{}_{}.pkl".format(study, class_aggregator, metric, label_type, drop_neg_control, aggregate_by_well, well_aggregator), "rb"))
    xlabels = ["CellProfiler", "DeepProfiler", "MOAProfiler"]
    entries = [CP, DP, MP]
    fig, ax = plt.subplots()
    width = .20
    x = np.array((range(1, len(xlabels) + 1)))
    ax.set_xticks(x)
    scores_map = {0: "F1", 1: "Precision", 2: "Recall"}
    color_map = {"F1": "grey", "Precision":"salmon", "Recall":"lightsteelblue"}
    for i in range(0, 3):
        score_type = scores_map[i]
        scores = [entries[j][i] for j in range(0, 3)]
        print("{} {} logistic regression {} {}, MP percent improvement over CP: {} and DP: {}".format(study, label_type, score_type, scores, (scores[2] - scores[0]) / scores[0], (scores[2] - scores[1]) / scores[1]))
        bar = ax.bar(x, scores, width=width, color=color_map[score_type], label=score_type)
        for i,j in zip(x, scores):
            ax.annotate("{:.0%}".format(j), xy=(i - .03, j +.03),fontsize=6)
        x = x + width 
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Score")
    ax.set_ylim((0,1.03))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', prop={"size":9}, bbox_to_anchor=(1, 1.32))
    plt.gcf().subplots_adjust(top=.76)
    plt.title("{}: Logistic Regression Performance Over Test Set".format(study.upper()))
    plt.savefig("outputs/logistic_regression_{}_{}_{}_{}_drop_neg_{}_well_aggregated_{}_{}_{}.png".format(class_aggregator, study, metric, label_type, drop_neg_control, aggregate_by_well, well_aggregator, deep_profile_type), dpi=300)


    
if os.path.isdir("outputs"):
    shutil.rmtree("outputs")
os.mkdir("outputs")

for study in ["JUMP1", "lincs"]:
    for date in os.listdir("save_dir/{}/multiclass_classification/".format(study)):
        plotLossCurves("save_dir/{}/multiclass_classification/{}/results.csv".format(study, date), plt_type="error")
        plotLossCurves("save_dir/{}/multiclass_classification/{}/results.csv".format(study, date), plt_type="loss")

csv_map = {
    "moa_targets_compounds": {"train": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_train.csv", "valid": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_valid.csv", "test": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test.csv", "test_no_neg": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test_no_negative.csv", "test_wells_excluded": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test_wells_excluded.csv", "full": "csvs/JUMP1/compound_images_with_MOA_no_polypharm_well_excluded.csv"},
    "moa_targets_compounds_polycompound": {"train": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_train.csv", "valid": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_valid.csv", "test": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test.csv", "test_no_neg": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test_wells_excluded_polycompound.csv", "test_wells_excluded": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test_wells_excluded_polycompound.csv", "full": "csvs/JUMP1/compound_images_with_MOA_no_polypharm_well_excluded.csv"},
    "moa_targets_compounds_holdout_2": {"train": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_well_excluded_compound_holdout_train_2_balanced_moas.csv", "valid": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_well_excluded_compound_holdout_valid_2.csv", "test": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_well_excluded_compound_holdout_test_2.csv",  "test_no_neg": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_well_excluded_compound_holdout_test_2_no_neg.csv", "test_wells_excluded": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_well_excluded_compound_holdout_test_2.csv", "full": "csvs/JUMP1/compound_images_with_MOA_no_polypharm_well_excluded.csv"},    
    "moa_targets_compounds_four_channel": {"train": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_train.csv", "valid": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_valid.csv", "test": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test.csv", "test_no_neg": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test_no_negative.csv", "test_wells_excluded": "csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_test_wells_excluded.csv", "full": "csvs/JUMP1/compound_images_with_MOA_no_polypharm_well_excluded.csv"},
    
    "moas_10uM": {"train": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_train.csv", "valid": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_valid.csv", "test": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_test.csv", "test_no_neg": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_test_no_negative.csv", "full": "csvs/lincs/lincs_ten_micromolar_no_polypharm.csv"}, 
    "moas_10uM_polycompound": {"train": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_train.csv", "valid": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_valid.csv" , "test": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_test_polycompound.csv" , "test_no_neg": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_test_polycompound.csv", "full": "csvs/lincs/lincs_ten_micromolar_no_polypharm.csv"}, 
    "moas_10uM_compounds_holdout_2": {"train": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_compound_holdout_train_2_balanced_moas.csv" , "valid": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_compound_holdout_valid_2.csv", "test": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_compound_holdout_test_2.csv", "test_no_neg": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_compound_holdout_test_2_no_neg.csv", "full": "csvs/lincs/lincs_ten_micromolar_no_polypharm.csv"}, 
    "moas_10uM_four_channel": {"train": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_train.csv", "valid": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_valid.csv", "test": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_test.csv", "test_no_neg": "csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_test_no_negative.csv", "full":  "csvs/lincs/lincs_ten_micromolar_no_polypharm.csv"}, 
}

permutations = [("JUMP1", "moa_targets_compounds_polycompound"), ("lincs", "moas_10uM_polycompound"), ("JUMP1", "moa_targets_compounds_holdout_2"), ("lincs", "moas_10uM_compounds_holdout_2")]

for study, label_type in permutations:
    if study == "JUMP1":
        deep_profile_types = ["model_{}".format(opt.well_aggregator)]
    else:
        deep_profile_types = ["model_{}".format(opt.well_aggregator), "bornholdt_trained"]
    
    ##plotReduction(deep_profile_type=deep_profile_types[0], study=study, label_type=label_type, aggregate_by_well=True, well_aggregator="median", method="TSNE", num_components=2)
    ##plotReducedColoredByPlate(study=study, label_type=label_type, aggregate_by_well=True, well_aggregator="median", method="TSNE", num_components=2, n_plates=5)

    if "compounds_holdout" in label_type:
        plotCompoundHoldoutKPredictionByLatentBars(study=study, label_type=label_type, deep_profile_type="model_{}".format(opt.well_aggregator), class_aggregator=opt.class_aggregator, metric=opt.metric)
        plotCompoundHoldoutMOABreakdown(study=study, label_type=label_type, deep_profile_type="model_{}".format(opt.well_aggregator), class_aggregator=opt.class_aggregator, metric=opt.metric)
        plotCompoundHoldoutKPrediction(study=study, label_type=label_type)
        generateIntraMOAvsInter(study=study, label_type=label_type, well_aggregator=opt.well_aggregator, metric=opt.metric)
        continue
        
    plotCellTypeTimepointSpecificPerformance(study=study, label_type=label_type)
    plotCompoundReplicateDistribution(study=study)
    plotFieldsPerWellDistribution(study=study)
    plotReplicateTrainingPerformances(study=study, label_type=label_type)
    plotMOATestSpread(study=study)
    plotClassificationMetrics(study=study, label_type=label_type) 
    plotClassSizeHistogram(study=study, verbose=False)
    plotMOAPertSpread(study=study)
    plotWellSpecificPerformance(study=study, label_type=label_type)
    generateIntraMOAvsInter(study=study, label_type=label_type, well_aggregator=opt.well_aggregator, metric=opt.metric)
    for deep_profile_type in deep_profile_types:
        print(deep_profile_type)
        plotDistributionByAggregatedLatentRep(study=study, class_aggregator=opt.class_aggregator, metric=opt.metric, label_type=label_type, drop_neg_control=True, aggregate_by_well=True, well_aggregator=opt.well_aggregator, deep_profile_type=deep_profile_type)
        plotLogisticRegression(study=study, class_aggregator=opt.class_aggregator, metric=opt.metric, label_type=label_type, drop_neg_control=True, aggregate_by_well=True, well_aggregator=opt.well_aggregator, deep_profile_type=deep_profile_type)
        plotEnrichment(study=study, class_aggregator=opt.class_aggregator, metric=opt.metric, label_type=label_type, drop_neg_control=True, aggregate_by_well=True, well_aggregator=opt.well_aggregator, deep_profile_type=deep_profile_type)
        plotScoreByAggregatedLatentRep(study=study, class_aggregator=opt.class_aggregator, metric=opt.metric, label_type=label_type, drop_neg_control=True, aggregate_by_well=True, well_aggregator=opt.well_aggregator, deep_profile_type=deep_profile_type)
        plotKNN(study=study, metric=opt.metric, label_type=label_type, drop_neg_control=True, aggregate_by_well=True, well_aggregator=opt.well_aggregator, deep_profile_type=deep_profile_type)
        plotReplicateAndNonreplicateSimilarity(study=study, metric=opt.metric, label_type=label_type, drop_neg_control=True, aggregate_by_well=True, well_aggregator=opt.well_aggregator, deep_profile_type=deep_profile_type)
        generateIntraPlatevsInter(study=study, label_type=label_type, deep_profile_type=deep_profile_type, well_aggregator=opt.well_aggregator, metric=opt.metric)


    