#
# main.py
# Jonathan Pilault, 2016-08-01
# Copyright (c) 2016 mldb.ai inc. All rights reserved.
#

import json
import re
import math
import main as mv



#mldb.log("Routes is called")
mldb = mldb_wrapper.wrap(mldb)
rp = mldb.plugin.rest_params


########################
# Variable definitions #
########################

# User defined
newMax = 1
newMin = 0
rad_train = 1.9
rad_user_img = 1.9
alpha_train = 10
alpha_user_img = 10
save_drawing = False


# Only change in models/main.py
mode = mv.get_mode()
allowable_models = mv.get_models()
unique_labels =  mv.get_allowed_labels()
base_model_url = "file://" + mldb.plugin.get_plugin_dir() + "/models/"

# No need to change
prob_scale = 100
explain_scale = 10
max_pixel_value = 255
min_pixel_value = 0
no_feat_drawing = 784
sq = 28 # square root of no_feat_drawing
dataset_url = "///mldb_data/datasets/"
save_name = "user_drawing"
final_name = "user_drawing_final"
final_dataset_name = "digits_raw" # digits_trans


########################
# Function definitions #
########################


def is_empty(anyDataStructure):
    if (anyDataStructure): return False
    else: return True

def centerNorm(row, mean, std):

    xms = zip(row, mean, std)
    new_row = map(lambda (x, m, s): (x-m) / s if abs(s)>0.001 else (x-m), xms)

    return new_row

def change_range(row, oldMax, oldMin, newMax, newMin):

    oldRange = oldMax - oldMin
    newRange = newMax - newMin
    new_row = map(lambda x: (x-oldMin) * newRange / oldRange + newMin, row)

    return new_row

def make_black_white(row):

    new_row = map(lambda x: 1 if abs(x)>0.1 else 0, row)

    return new_row

def create_convolution():

    JsConvolutionExpr = """
        jseval('
            var row_val = val;
            var radius = rad;
            var alpha = alpha_;

            // input 1D list, output 1D list, width, height, alpha
            function laplaceSharpen(inList, outList, w, h, alpha) {

                var weights = [1, 1, 1, 1, -8, 1, 1, 1, 1]; // sharpen convolution matrix
                var rs = 3; // kind of similar to radius -- needs to be sqrt(weights.length)

                for (var i = 0; i < h; i++)
                    for (var j = 0; j < w; j++) {
                        var val = 0;
                        var indexW = 0;
                        for (var iy = i; iy < i + rs; iy++)
                            for (var ix = j; ix < j + rs; ix++) {
                              var x = Math.min(w - 1, Math.max(0, ix));
                              var y = Math.min(h - 1, Math.max(0, iy));
                              val += inList[y * w + x] * weights[indexW];
                              indexW ++;
                            }
                        new_value = inList[i * w + j] - val * alpha;
                        outList[i * w + j] = new_value;
                    }
                return outList;
            } // End of laplaceSharpen

            // input 1D list, output 1D list, width, height, radius
            function gaussianBlur(inList, outList, w, h, r) {

                var rs = Math.ceil(r * 2.57); // index around pixel must be int
                for (var i = 0; i < h; i++)
                    for (var j = 0; j < w; j++) {
                        var val = 0,
                        wsum = 0;
                        for (var iy = i - rs; iy < i + rs + 1; iy++)
                            for (var ix = j - rs; ix < j + rs + 1; ix++) {
                                var x = Math.min(w - 1, Math.max(0, ix));
                                var y = Math.min(h - 1, Math.max(0, iy));
                                var dsq = (ix - j) * (ix - j) + (iy - i) * (iy - i);
                                var wght = Math.exp(-dsq / (2 * r * r)) / (Math.PI * 2 * r * r);
                                val += inList[y * w + x] * wght;
                                wsum += wght;
                            }
                        outList[i * w + j] = val / wsum;
                    }
                return outList;
            } // End of gaussianBlur


            //Assuring that the 1d row is in the right order
            var length = row_val.length;
            var dim = Math.sqrt(length);
            var matrix = new Array(length);
            for (var i = 0; i < length; i++) {
                matrix[row_val[i][0][0]] = row_val[i][1];
            }

            //Using functions
            var blurredMatrix = [];
            var NoisyMatrix = [];
            blurredMatrix = gaussianBlur(matrix, blurredMatrix, dim, dim, radius);
            NoisyMatrix = laplaceSharpen(blurredMatrix, NoisyMatrix, dim, dim, alpha);

            return NoisyMatrix;

            ','val, rad, alpha_', valueExpr, radius, alpha
        ) AS *
    """


    print mldb.put("/v1/functions/convolution", {
        "type": "sql.expression",
        "params": {
            "expression": JsConvolutionExpr,
            "prepared": True
        }
    })




def get_explain_and_prob(valueExpr, radius, alpha, unique_labels, procedureRunName,
        no_feat_drawing):


    probExpr = ""
    explainExpr = ""
    explains = []
    probabilities = []
    for i in unique_labels:
        probExpr = probExpr + \
        "probabilizer_%(procedureRunName)s_%(i)d({score: scores.\"%(i)d\"})[prob] as prob_%(i)d, " \
        %{"procedureRunName": procedureRunName,
          "i" : i}
        explainExpr = explainExpr + \
        "explain_%(procedureRunName)s({features: %(valueExpr)s, label: %(i)d})[explanation] AS explain_%(i)d, " \
        %{"procedureRunName": procedureRunName,
          "i" : i,
          "valueExpr": valueExpr
          }


    SQL_Expr = """
        SELECT
            %(probExpr)s
            explain*
        FROM (
            SELECT
                %(explainExpr)s
                %(procedureRunName)s_scorer_0({features: %(valueExpr)s})[scores] AS scores
            )
    """ %  {
            "probExpr": probExpr,
            "explainExpr": explainExpr,
            "procedureRunName": procedureRunName,
            "valueExpr": valueExpr,
            "radius": radius,
            "alpha": alpha
            }

    queryDump = mldb.query(SQL_Expr)

    #mldb.log(queryDump)


    target = -1

    for resultHeaders, results in zip(queryDump[0], queryDump[1]):
        explainMatch = re.match('(?:explain_)(_*.)(_*.)(.*.)', resultHeaders)
        probMatch = re.match('(?:prob_)(_*.)', resultHeaders)

        if (explainMatch != None):
            current_target = int(explainMatch.group(1))
            index = int(explainMatch.group(3))

            if current_target != target:
                explains.append([0] * no_feat_drawing)
                target = current_target


            explains[current_target][index] = float(results) * explain_scale

            #mldb.log(explain[index])
            #mldb.log(explains[current_target][index])

        elif (probMatch != None):
            probabilities.append(float(results) * prob_scale)


    return (explains, probabilities)



def save_to_csv(drawing_features, no_feat_drawing, label_names, label_values,
        path, save_name):

    import csv
    import os

    file_name = path + save_name + ".csv"

    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

    with open(file_name, 'wb') as ds:
        wr = csv.writer(ds, quoting=csv.QUOTE_ALL)
        header = [str(i) for i in range(no_feat_drawing)]
        if (label_names != None):
            header.extend(label_names)
        wr.writerow(header)
        if (label_values != None):
            for feature, label_value in zip(drawing_features, label_values):
                feature.extend(label_value)
                wr.writerow(feature)
        else:
            wr.writerow(drawing_features)

    conf = {
        "type":"import.text",
        "params": {
            "dataFileUrl": "file:" + file_name,
            "outputDataset": {
                "id": "%s" %save_name,
                "type": "tabular"
            },
            "runOnCreation": True
        }
    }
    create_drawingSave = mldb.put("/v1/procedures/%s" % save_name, conf)
    mldb.log(create_drawingSave)



def addNoise(data_name, final_dataset_name, labelNeeded, radius, alpha):

    # This steps does the equivalent of gaussianBlur and laplaceSharpen
    # It is faster since it uses MLDB's parallel processing capabilities

    mldb.log("data_name: %s" %data_name)
    mldb.log("final_dataset_name: %s" %final_dataset_name)
    mldb.log("labelNeeded: %s" %labelNeeded)
    mldb.log("radius: %s" %radius)
    mldb.log("alpha: %s" %alpha)



    valueExpr = "{*}"
    labelsExpr = ""

    if labelNeeded: # Labels needed in training/testing data creation step
        valueExpr = "{* EXCLUDING(label*)}"
        labelsExpr = ",label*"

    SQL_Expr = """
        SELECT convolution({valueExpr: %(valueExpr)s, radius: %(radius)d,
                            alpha: %(alpha)d}) AS *
        %(labelsExpr)s
        FROM %(datasetName)s
    """ %   {
                "valueExpr": valueExpr,
                "radius": radius,
                "alpha": alpha,
                "labelsExpr": labelsExpr,
                "datasetName": data_name
            }

    conf = {
        "type": "transform",
        "params": {
            "inputData": SQL_Expr,
            "outputDataset": {
                "id": "%s" %final_dataset_name,
                "type": "tabular"
            },
            "runOnCreation": True
        }
    }

    create_noisy_data =  mldb.put('/v1/procedures/%s' % final_dataset_name, conf)
    mldb.log(create_noisy_data)



###############################
###### Payload handling #######
###############################

if rp.verb == "GET" and rp.remaining == "/mnist_pics":

    import random
    import png

    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #pic_array = []

    colors = [
            [0, 0, 0, 0],
            [0, 0, 0, 128], # [0, 0, 255, 128] for transparent bleu
            ]
    no_examples = 11000

    for label in labels:
        offset = random.randint(0, no_examples/len(labels))
        x_data = mldb.query("""
            SELECT * EXCLUDING(label)
            FROM digits_mnist
            WHERE label = %d
            LIMIT 1
            OFFSET %d
        """ %(label, offset))

        pixels = make_black_white(row = x_data[1][1:]) # pixels here is either 0 or 1
        pixels = sum(map(lambda x: colors[x], pixels), []) # pixels is converted in colors
        file_name = mldb.plugin.get_plugin_dir() + "/static/%d.png" %label

        f = open(file_name, 'wb')      # binary mode is important
        w = png.Writer(width=sq, height=sq, alpha=True)
        w.write_array(f, pixels)
        f.close()

    mldb.plugin.set_return("");


###############################
###### Payload handling #######
###############################

if rp.verb == "POST" and rp.remaining == "/handle_drawing":


    ########################
    #  Input payload info  #
    ########################

    payload = json.loads(rp.payload)
    model = payload['procedure']
    drawing_features = payload['user_input']

    ########################
    # Setting up variables #
    ########################

    accuracy = None
    explainString = None
    explains = []
    probabilities = []
    modelFileUrlPattern = base_model_url + model


    if (model not in allowable_models):
        mldb.log("Chosen procedure was not created yet!!")

    else:


        drawing_features = change_range(row = drawing_features, oldMax = max_pixel_value,
            oldMin = min_pixel_value, newMax = newMax, newMin = newMin)

        if save_drawing:
            save_to_csv(drawing_features = drawing_features, no_feat_drawing =  no_feat_drawing,
            label_names = None, label_values = None,
            path=dataset_url, save_name=save_name)


        _model = "_" + model
        _mode = "_" + mode
        procedureRunName = "mnist" + _model + _mode

        create_convolution()

        explains, probabilities = get_explain_and_prob(valueExpr = drawing_features, radius = rad_user_img,
            alpha = alpha_user_img, unique_labels = unique_labels, procedureRunName = procedureRunName,
            no_feat_drawing = no_feat_drawing)



    output_ = {
        "accuracy": accuracy,
        "explainString": explainString,
        "explains": explains,
        "scores": probabilities
    }

    mldb.plugin.set_return(output_);



#######################################
############ DATA CREATION ############
#######################################

if rp.remaining == "/create_data":


    ########################
    # Variable definitions #
    ########################

    # Do not change
    mnist_data_url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'

    # User defined

    data_name = "digits_raw"
    label_name = "label"

    mldb.log(mldb.plugin.get_plugin_dir())
    mldb.log("Main.py is loaded to create the MNIST dataset...")

    import cPickle, gzip
    from StringIO import StringIO
    from urllib import urlopen
    import numpy as np


    sets = []

    # Load the data
    try:
        inmemory = StringIO(urlopen(mnist_data_url).read())
        fStream = gzip.GzipFile(fileobj=inmemory, mode='rb')
        sets = list(cPickle.load(fStream)) #tupple has 3 sets: train, validation and test sets
        fStream.close()

    except Exception as xcpt:
        mldb.log("Failed to load file %s" % xcpt)

    if (is_empty(sets)):
        pass

    else:
        mldb.delete("/v1/datasets/%s" % data_name)

        preprocessedData = []
        labelNames = []
        labelData = []
        index = 0
        mean = 0
        std = 1

        for i, set_ in enumerate(sets):


            if (i == 0):
                labelNames.append(label_name)
                for target in enumerate(unique_labels):
                    labelNames.append(label_name + "_" + str(target[1]))

                #mean = np.mean(set_[0], axis=0) # doing this only for train set is good enough
                #std = np.std(set_[0], axis=0) # doing this only for train set is good enough

            # each set is composed of a feature matrix and labels
            data = set_[0].tolist()
            label = set_[1].tolist()



            for j, row in enumerate(data):

                ###################
                # Preprocess data #
                ###################

                # Zero-centering and normalize data
                #row = centerNorm(row = row, mean = mean, std = std)

                #oldMax = np.amax(row)
                #oldMin = np.amin(row)

                # Change range --> i.e. linear models don't like zero values
                #row = change_range(row = row, oldMax = oldMax, oldMin = oldMin,
                    #newMax = newMax, newMin = newMin)

                row = make_black_white(row = row)

                labelRow = [label[j]]
                # One hot vectors for each label:
                for target in enumerate(unique_labels):
                    bool_value = 0
                    if target[1] == label[j]: bool_value = 1
                    labelRow.append(bool_value)

                # appending newest row
                preprocessedData.append(row)
                labelData.append(labelRow)

            # End for loop

            index = index + 1
        # End for loop

        # Creatng a dataset via a csv MLDB text import procedure
        save_to_csv(drawing_features = preprocessedData, no_feat_drawing = no_feat_drawing,
            label_names = labelNames, label_values = labelData,
            path = dataset_url, save_name = data_name)
        mldb.log("Saved to CSV and initial dataset created.")

        # creating convolution function to use in the next step
        #create_convolution()

        # Adding noise by blurring and sharpening images
        #addNoise(data_name = data_name, final_dataset_name = final_dataset_name,
            #labelNeeded = True, radius = rad_train, alpha = alpha_train)
        #mldb.log("Initial dataset transformed with added noise.")

    mldb.plugin.set_return("Success in creating data");


########################################
############ MODEL CREATION ############
########################################

if rp.remaining == "/train_models":

    data_name = final_dataset_name
    label_name = "label"



    mldb.log("Main.py is loaded to train classifier and probabilizer models...")


    # Configuration dictionary:
    bbdt_d5_config = {
        "bbdt_d5": {
            "type": "bagging",
            "verbosity": 3,
            "weak_learner": {
                "type": "boosting",
                "verbosity": 3,
                "weak_learner": {
                    "type": "decision_tree",
                    "max_depth": 7,
                    "verbosity": 0,
                    "update_alg": "gentle",
                    "random_feature_propn": 0.3
                },
                "min_iter": 5,
                "max_iter": 30
            },
            "num_bags": 5
        }
    }

    dt_config = {
        "dt": {
            "type": "decision_tree",
            "max_depth": 14,
            "verbosity": 3,
            "update_alg": "prob"
            ,"random_feature_propn": 0.8
        }
    }

    glz_config = {
        "glz": {
            "type": "glz",
            "verbosity": 3,
            "normalize": True,
            "link_function": "logit",
            "regularization": "l2"
        }
    }

    bglz_config = {
        "bglz": {
            "_note": "Bagged random GLZ",
            "type": "bagging",
            "verbosity": 1,
            "validation_split": 0.1,
            "weak_learner": {
                "type": "glz",
                "feature_proportion": 1.0,
                "link_function": "logit",
                "regularization": "l2",
                "verbosity": 0
            },
            "num_bags": 6
        }
    }

    bbs2_config = {
        "bbs2": {
            "_note": "Bagged boosted stumps",

            "type": "bagging",
            "num_bags": 5,
            "weak_learner": {
                "type": "boosting",
                "verbosity": 3,
                "weak_learner": {
                    "type": "decision_tree",
                    "max_depth": 1,
                    "verbosity": 0,
                    "update_alg": "gentle"
                },
                "min_iter": 5,
                "max_iter": 300,
                "trace_training_acc": "true"
            }
        }
    }

    configs_dct = {
        "bbdt_d5": bbdt_d5_config,
        "dt": dt_config,
        "glz": glz_config,
        "bglz": bglz_config,
        "bbs2": bbs2_config
    }


    for model in allowable_models:

        config = configs_dct[model]
        modelFileUrlPattern = base_model_url + model
        run_again = True

        for target in unique_labels:

            _model = "_" + model
            _mode = "_" + mode

            procedureRunName = "mnist" + _model + _mode

            if (mode == "categorical"):
                _target = ""
            if (mode == "boolean"):
                _target = "_" + str(target)
                run_again = True # "boolean" runs for each target while categorical runs only once

            if (run_again):

                # classifier model
                a = mldb.delete("/v1/procedures/%s" % (procedureRunName + _target))
                a_f = mldb.delete("/v1/functions/%s" % (procedureRunName + _target))
                a_s = mldb.delete("/v1/functions/%s_scorer_0" % (procedureRunName + _target))
                mldb.log("DELETE classifier")
                mldb.log(a)
                conf_class = {
                    "type": "classifier.experiment",
                    "params": {
                        "experimentName": (procedureRunName + _target),
                        "mode": mode,
                        "inputData" : """
                            select
                                {* EXCLUDING(label*)} AS features,
                                label%s AS label
                            from %s
                        """ % (_target, data_name),
                        "datasetFolds": [
                            {
                                "trainingWhere": "rowHash() % 5 != 0",
                                "testingWhere": "rowHash() % 5 = 0"

                                #,"trainingLimit": 500,
                                #"testingLimit": 50
                            }],
                        "algorithm": model,
                        "configuration": config,
                        "modelFileUrlPattern": "%s/%s.cls" % (modelFileUrlPattern, (procedureRunName + _target)),
                        "keepArtifacts": True,
                        "outputAccuracyDataset": True,
                        "runOnCreation": True,
                        "evalTrain": True
                    }
                }
                mldb.log(" TRAINING %s CLASSIFIER **********************" % model)
                b = mldb.put("/v1/procedures/%s" % (procedureRunName + _target), conf_class)
                mldb.log("PUT classifier")
                mldb.log(b)

                run_again = False # "boolean" runs for each target while categorical runs only once


            _target_prob = "_" + str(target)

            # probabilizer model
            c = mldb.delete("/v1/procedures/probabilizer_%s" % (procedureRunName + _target_prob))
            mldb.log("DELETE probabilizer")
            mldb.log(c)
            conf_prob = {
                "type": "probabilizer.train",
                "params": {
                    "trainingData":{
                            "select": """
                                %(path)s_scorer_0({ features: {* EXCLUDING(label*)} })[scores.\"%(target)s\"] AS score,
                                label%(_target_prob)s AS label
                            """
                            % {"path":(procedureRunName + _target), "target": str(target), "_target_prob":_target_prob},
                            "from": {"id": data_name},
                            "where": "rowHash() % 5 = 0" # this should be trained on the test set
                    },
                    "modelFileUrl": "%s/probabilizer_%s.cls" % (modelFileUrlPattern, (procedureRunName + _target_prob)),
                    "functionName": "probabilizer_%s" % (procedureRunName + _target_prob),
                    "runOnCreation": True
                }
            }
            mldb.log(" TRAINING %s PROBABILIZER **********************" % model)
            d = mldb.put("/v1/procedures/probabilizer_%s" % (procedureRunName + _target_prob), conf_prob)
            mldb.log("PUT probabilizer")
            mldb.log(d)

    mldb.plugin.set_return("Success in training models");
