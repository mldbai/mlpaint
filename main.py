#
# main.py
# Jonathan Pilault, 2016-08-01
# Copyright (c) 2016 Datacratic Inc. All rights reserved.
#

########################
# Variable definitions #
########################

# Only change in main.py
mode = "categorical" # "boolean", "categorical"
allowable_models = ["bbdt_d5", "bbs2", "bglz", "dt"] # ["bbdt_d5", "bbs2", "bglz", "dt", "glz"]
unique_labels = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]

########################
# Function definitions #
########################

def get_mode():
    global mode
    return mode

def get_models():
    global allowable_models
    return allowable_models

def get_allowed_labels():
    global unique_labels
    return unique_labels


########################
########  Main  ########
########################


if __name__ == '__main__':

    mldb.plugin.serve_static_folder('/static', 'static')
    mldb = mldb_wrapper.wrap(mldb)

    base_model_url = "file://" + mldb.plugin.get_plugin_dir() + "/models/"
    mldb.log("Loading Main.py...")


    # Get test set for sending pics in the front-end

    data_url_mnist = 'http://public.mldb.ai/datasets/digits_data.csv.gz'

    print mldb.put('/v1/procedures/import_digits_mnist', {
        "type":"import.text",
        "params": {
            "dataFileUrl": data_url_mnist,
            "outputDataset": "digits_mnist",
            "select": "{* EXCLUDING(\"785\")} AS *, \"785\" AS label",
            "runOnCreation": True,
            "where": "rowHash() % 5 = 0"
        }
    })



    for model in allowable_models:

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

            mldb.log("********************* %s *********************" % (procedureRunName + _target))

            if (run_again):

                #############################
                #   Check/Create Explain    #
                #############################

                try:
                    conf_explain = {
                        "id": "explain_%s" % (procedureRunName + _target),
                        "type": "classifier.explain",
                        "params": { "modelFileUrl": "%s/%s.cls" % (modelFileUrlPattern, (procedureRunName + _target))}
                    }
                    create_explain = mldb.put("/v1/functions/explain_%s" % (procedureRunName + _target), conf_explain)
                    mldb.log(create_explain)
                    mldb.log("explain_%s created" % (procedureRunName + _target))

                except Exception as xcpt:
                    mldb.log(xcpt)


                #############################
                #    Check/Create Scorer    #
                #############################

                try:
                    conf_score = {
                        "id": "%s_scorer_0" % (procedureRunName + _target),
                        "type": "classifier",
                        "params": { "modelFileUrl": "%s/%s.cls" % (modelFileUrlPattern, (procedureRunName + _target))}
                    }
                    create_score = mldb.put("/v1/functions/%s_scorer_0" % (procedureRunName + _target), conf_score)
                    mldb.log(create_score)
                    mldb.log("%s_scorer_0" % (procedureRunName + _target))

                    run_again = False # "boolean" runs for each target while categorical runs only once

                except Exception as xcpt:
                    mldb.log(xcpt)


            #############################
            # Check/Create Probabilizer #
            #############################


            _target_prob = "_" + str(target)


            try:
                conf_proba = {
                    "id": "probabilizer_%s" % (procedureRunName + _target_prob),
                    "type": "probabilizer",
                    "params": { "modelFileUrl": "%s/probabilizer_%s.cls" % (modelFileUrlPattern, (procedureRunName + _target_prob))}
                }
                create_probabilizer = mldb.put("/v1/functions/probabilizer_%s" % (procedureRunName + _target_prob), conf_proba)
                mldb.log(create_probabilizer)

            except Exception as xcpt:
                mldb.log(xcpt)


