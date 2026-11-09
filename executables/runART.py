#!/usr/bin/env python3
"""Train or load an ART model from a YAML config and write the ART-ready training CSV."""


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import argparse

import pandas as pd
import cloudpickle
import yaml

sys.path.append(".")
sys.path.append("<ART_PATH>")

try:
    import warning_utils
    warning_utils.filter_end_user_warnings()
except ModuleNotFoundError:
    print("warning_utils not found; continuing without custom warning filtering.")

from art.core import RecommendationEngine
import art.utility as utils


def parseArgs():
    """Parse the command line, which takes the path to the YAML config file."""

    parser = argparse.ArgumentParser(
        description="Run ART workflow using a YAML configuration file."
    )
    parser.add_argument(
        "configFile",
        help="Path to YAML config file. Example: python runART.py artConfig.yaml"
    )
    return parser.parse_args()


def loadConfig(configFile):
    """Read the YAML config and raise ValueError if it is empty."""

    with open(configFile, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Config file is empty.")

    return config


def getRequiredConfig(config, keys):
    """Raise ValueError listing any of the given keys that are missing from the config."""

    missingKeys = [key for key in keys if key not in config]
    if missingKeys:
        raise ValueError(f"Missing required config keys: {missingKeys}")


def buildPaths(config):
    """Resolve the run, data and output paths from the config.

    Creates the output directory. Explicit inputTrainingCsv, artReadyCsv and
    artModelFile entries override the defaults built from jobName and the
    configured suffixes.
    """

    runDir = os.getcwd()

    baseDataDir = config["baseDataDir"]
    modelBuildingDataSubdir = config.get("modelBuildingDataSubdir", ".")
    modelBuildingDataDir = os.path.join(baseDataDir, modelBuildingDataSubdir)

    outputDir = os.path.join(runDir, config["outputDir"])
    os.makedirs(outputDir, exist_ok=True)

    jobName = config["jobName"]

    defaultTrainingCsvName = f"{jobName}{config['trainingCsvSuffix']}"
    defaultArtReadyCsvName = f"{jobName}{config['artReadyCsvSuffix']}"

    inputTrainingCsv = config.get("inputTrainingCsv") or os.path.join(
        modelBuildingDataDir, defaultTrainingCsvName
    )

    artReadyCsv = config.get("artReadyCsv") or os.path.join(
        outputDir, defaultArtReadyCsvName
    )

    artModelFile = config.get("artModelFile") or os.path.join(
        outputDir, config["artModelFileName"]
    )

    return {
        "runDir": runDir,
        "baseDataDir": baseDataDir,
        "modelBuildingDataDir": modelBuildingDataDir,
        "outputDir": outputDir,
        "inputTrainingCsv": inputTrainingCsv,
        "artReadyCsv": artReadyCsv,
        "artModelFile": artModelFile,
    }


def loadTrainingData(config, paths):
    """Read the training CSV, keeping only the first limitTrainingRows rows if that is set."""

    print("Loading training data...")
    DF = pd.read_csv(paths["inputTrainingCsv"])

    rowLimit = config.get("limitTrainingRows")
    if rowLimit is not None:
        DF = DF.iloc[:rowLimit].copy()
        print(f"Using first {len(DF)} rows from training data")

    return DF


def getInputAndResponseColumns(config, DF):
    """Return the feature columns (those starting with the configured prefix) and the response columns.

    Raises ValueError if no feature column matches or a response column is missing.
    """

    inputColumnPrefix = config["inputFeatureColumnPrefix"]
    responseColumns = config["responseColumns"]

    inputColumns = [col for col in DF.columns if col.startswith(inputColumnPrefix)]

    if len(inputColumns) == 0:
        raise ValueError(
            f"No input feature columns found with prefix '{inputColumnPrefix}'."
        )

    missingResponseColumns = [col for col in responseColumns if col not in DF.columns]
    if missingResponseColumns:
        raise ValueError(
            f"Missing response columns in training data: {missingResponseColumns}"
        )

    print(f"Detected {len(inputColumns)} input feature columns")
    print(f"Response columns: {responseColumns}")

    return inputColumns, responseColumns


def saveArtReadyCsv(trainingDF, inputColumns, responseColumns, paths):
    """Write the training data in the EDD format that ART reads, then reload and return it."""

    print(f"Saving ART-ready CSV to: {paths['artReadyCsv']}")

    features = trainingDF[inputColumns].to_numpy()
    response = trainingDF[responseColumns].to_numpy()

    utils.save_edd_csv(
        features,
        response,
        inputColumns,
        paths["artReadyCsv"],
        responseColumns
    )

    print("Reloading ART-ready CSV...")
    artReadyDF = pd.read_csv(paths["artReadyCsv"])
    return artReadyDF


def loadArtModel(paths):
    """Load a saved ART model from the path in paths['artModelFile']."""

    print(f"Loading ART model from: {paths['artModelFile']}")
    with open(paths["artModelFile"], "rb") as f:
        artModel = cloudpickle.load(f)
    return artModel


def trainArtModel(config, artReadyDF, inputColumns, responseColumns, paths):
    """Train a RecommendationEngine from the config and return it.

    The model is saved with cloudpickle if saveTrainedArtModel is true.
    """

    artParams = {
        "input_vars": inputColumns,
        "response_vars": responseColumns,
        "objective": config["artObjective"],
        "threshold": config["artThreshold"],
        "alpha": config["artAlpha"],
        "num_recommendations": config["artNumRecommendations"],
        "max_mcmc_cores": config["artMaxMcmcCores"],
        "seed": config["randomSeed"],
        "output_dir": paths["outputDir"],
        "recommend": config["artGenerateRecommendations"],
        "cross_val": config["artRunCrossValidation"],
        "num_tpot_models": config["artNumTpotModels"],
    }

    print("Training ART model...")
    startTime = time.time()
    artModel = RecommendationEngine(df=artReadyDF, **artParams)
    elapsedTime = time.time() - startTime
    print(f"ART training completed in {elapsedTime:.2f} seconds")

    if config["saveTrainedArtModel"]:
        with open(paths["artModelFile"], "wb") as f:
            cloudpickle.dump(artModel, f)
        print(f"Saved ART model to: {paths['artModelFile']}")

    return artModel


def getArtModel(config, artReadyDF, inputColumns, responseColumns, paths):
    """Load the existing ART model if useExistingArtModel is true, otherwise train a new one."""

    if config["useExistingArtModel"]:
        return loadArtModel(paths)

    return trainArtModel(config, artReadyDF, inputColumns, responseColumns, paths)


def main():
    """Run the ART workflow from the command line and return the ART model."""

    startTimeTotal = time.time()

    args = parseArgs()
    config = loadConfig(args.configFile)

    requiredKeys = [
        "baseDataDir",
        "modelBuildingDataSubdir",
        "outputDir",
        "jobName",
        "trainingCsvSuffix",
        "artReadyCsvSuffix",
        "artModelFileName",
        "inputFeatureColumnPrefix",
        "responseColumns",
        "useExistingArtModel",
        "saveTrainedArtModel",
        "artObjective",
        "artThreshold",
        "artAlpha",
        "artNumRecommendations",
        "artMaxMcmcCores",
        "artGenerateRecommendations",
        "artRunCrossValidation",
        "artNumTpotModels",
        "randomSeed",
    ]
    getRequiredConfig(config, requiredKeys)

    paths = buildPaths(config)

    print("Resolved paths:")
    for key, value in paths.items():
        print(f"  {key}: {value}")

    trainingDF = loadTrainingData(config, paths)
    inputColumns, responseColumns = getInputAndResponseColumns(config, trainingDF)
    artReadyDF = saveArtReadyCsv(trainingDF, inputColumns, responseColumns, paths)
    artModel = getArtModel(config, artReadyDF, inputColumns, responseColumns, paths)

    elapsedTimeTotal = time.time() - startTimeTotal
    print(f"Total time: {elapsedTimeTotal:.2f} seconds")
    print(f"Job complete ...")

    return artModel


if __name__ == "__main__":
    main()