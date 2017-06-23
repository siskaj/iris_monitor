#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Model training for Iris data set using Validation Monitor."""


import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib as contrib
import pandas as pd

from datetime import datetime

COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
LABEL = "species"

tf.logging.set_verbosity(tf.logging.INFO)

# Data sets
IRIS_TRAINING = os.path.join(os.path.dirname(__file__), "iris_training.csv")
IRIS_TEST = os.path.join(os.path.dirname(__file__), "iris_test.csv")

def input_fn(df):
    feature_cols = {k: tf.constant(df[k].values) for k in FEATURES}
    labels = tf.constant(df[LABEL].values)
    return feature_cols, labels

def main(unused_argv):
    # Load datasets.
    training_set = contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING, target_dtype=np.int, features_dtype=np.float)
    test_set = contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST, target_dtype=np.int, features_dtype=np.float)

    df_train = pd.read_csv("iris_training.csv", names=COLUMNS, skiprows=1)
    df_test = pd.read_csv("iris_test.csv", names=COLUMNS, skiprows=1)

    feature_cols = [contrib.layers.real_valued_column(k) for k in FEATURES]

    validation_metrics = {
        "accuracy":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_accuracy,
                prediction_key="classes"),
        "precision":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_precision,
                prediction_key="classes"),
        "recall":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_recall,
                prediction_key="classes")
    }
    validation_monitor = contrib.learn.monitors.ValidationMonitor(
        # test_set.data,
        # test_set.target,
        input_fn = lambda: input_fn(df_test),
        every_n_steps=50,
        metrics=validation_metrics,
        eval_steps=1,
        early_stopping_metric="loss",
        early_stopping_metric_minimize=True,
        early_stopping_rounds=200
    )

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "/tmp/iris_model"
    logdir = "{}/run-{}/".format(root_logdir, now)
    # Specify that all features have real-value data
#    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = contrib.learn.DNNClassifier(
        feature_columns=feature_cols,
        hidden_units=[10, 20, 10],
        n_classes=3,
        model_dir=logdir,
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))

    # Fit model.
    # classifier.fit(x=training_set.data,
    #                y=training_set.target,
    #                steps=2000,
    #                monitors=[validation_monitor])

    classifier.fit(input_fn = lambda : input_fn(df_train),
                   steps=2000,
                   monitors=[validation_monitor])

    # classifier.fit(input_fn = lambda : input_fn(df_train),
    #                steps=2000
    #                )

    # Evaluate accuracy.
    # accuracy_score = classifier.evaluate(
    #     x=test_set.data, y=test_set.target)["accuracy"]
    # print("Accuracy: {0:f}".format(accuracy_score))

    print("Jdu na Evaluation")
    accuracy_score = classifier.evaluate(input_fn = lambda : input_fn(df_test), steps=1)["accuracy"]
    print("Accuracy: {0:f}".format(accuracy_score))

    # Classify two new flower samples.
    new_samples = np.array(
        [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
#    y = list(contrib.learn.SKCompat(classifier.predict_classes(new_samples)))
    y = list(classifier.predict(new_samples))
    print("Predictions: {}".format(str(y)))


if __name__ == "__main__":
    tf.app.run()
