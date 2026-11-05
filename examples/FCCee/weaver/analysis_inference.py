## input file needed for unit test in CI
testFile = "https://fccsw.web.cern.ch/fccsw/testsamples/wzp6_ee_nunuH_Hss_ecm240.root"

## output directory
outputDir   = "outputs/inference"

## pre_summer2026 7-class (G/U/D/S/C/B/TAU) particle transformer, 70M jets
model_dir = "/gluster/home/jspittle/ML_model"
model_name = "fccee_flavtagging_edm4hep_wc_v1"

weaver_preproc = "{}/{}.json".format(model_dir, model_name)
weaver_model = "{}/{}.onnx".format(model_dir, model_name)

from addons.ONNXRuntime.jetFlavourHelper import JetFlavourHelper
from addons.FastJet.jetClusteringHelper import ExclusiveJetClusteringHelper

jetFlavourHelper = None
jetClusteringHelper = None

# Mandatory: RDFanalysis class where the use defines the operations on the TTree
class RDFanalysis:
    # __________________________________________________________
    # Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2
    def analysers(df):
        global jetClusteringHelper
        global jetFlavourHelper

        from examples.FCCee.weaver.config import collections, njets

        tag = ""

        ## define jet clustering parameters
        jetClusteringHelper = ExclusiveJetClusteringHelper(collections["PFParticles"], njets, tag)

        ## run jet clustering
        df = jetClusteringHelper.define(df)

        ## define jet flavour tagging parameters

        jetFlavourHelper = JetFlavourHelper(
            collections,
            jetClusteringHelper.jets,
            jetClusteringHelper.constituents,
            tag,
        )

        ## define observables for tagger
        df = jetFlavourHelper.define(df)

        df = df.Define("event_id", "rdfentry_")
        df = df.Filter("event_id % 5 == 0")

        ## tagger inference
        df = jetFlavourHelper.inference(weaver_preproc, weaver_model, df)

        return df

    # __________________________________________________________
    # Mandatory: output function, please make sure you return the branchlist as a python list
    def output():

        ##  outputs jet properties
        branchList = jetClusteringHelper.outputBranches()

        ## outputs jet scores and constituent breakdown
        branchList += jetFlavourHelper.outputBranches()

        return branchList
