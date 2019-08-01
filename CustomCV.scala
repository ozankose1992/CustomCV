/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.ml.tuning // !!!Important

import org.apache.spark.annotation.Since
import org.apache.spark.ml.Model
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.Dataset
import org.apache.spark.util.ThreadUtils

import scala.concurrent.Future
import scala.concurrent.duration.Duration


/**
 * K-fold cross validation performs model selection by splitting the dataset into a set of
 * non-overlapping randomly partitioned folds which are used as separate training and test datasets
 * e.g., with k=3 folds, K-fold cross validation will generate 3 (training, test) dataset pairs,
 * each of which uses 2/3 of the data for training and 1/3 for testing. Each fold is used as the
 * test set exactly once.
 */
@Since("1.2.0")
class CustomCV @Since("1.2.0") (@Since("1.4.0") override val uid: String)
  extends CrossValidator {

  @Since("1.2.0")
  def this() = this(Identifiable.randomUID("cv"))

  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): CrossValidatorModel = instrumented { instr =>
    val schema = dataset.schema
    transformSchema(schema, logging = true)
    val sparkSession = dataset.sparkSession
    val est = $(estimator)
    val eval = $(evaluator)
    val epm = $(estimatorParamMaps)

    // Create execution context based on $(parallelism)
    val executionContext = getExecutionContext

    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, numFolds, seed, parallelism)
    logTuningParams(instr)

    val collectSubModelsParam = $(collectSubModels)

    var subModels: Option[Array[Array[Model[_]]]] = if (collectSubModelsParam) {
      Some(Array.fill($(numFolds))(Array.fill[Model[_]](epm.length)(null)))
    } else None

    // Compute metrics for each model over each split
    val splits = MLUtils.kFold(dataset.toDF.rdd, $(numFolds), $(seed))
    val metrics = splits.zipWithIndex.map { case ((training, validation), splitIndex) =>
      val trainingDataset = sparkSession.createDataFrame(training, schema).cache()
      val validationDataset = sparkSession.createDataFrame(validation, schema).cache()
      instr.logDebug(s"Train split $splitIndex with multiple sets of parameters.")

      // Fit models in a Future for training in parallel
      val foldMetricFutures = epm.zipWithIndex.map { case (paramMap, paramIndex) =>
        Future[Double] {
          val model = est.fit(trainingDataset, paramMap).asInstanceOf[Model[_]]
          if (collectSubModelsParam) {
            subModels.get(splitIndex)(paramIndex) = model
          }
          // TODO: duplicate evaluator to take extra params from input
          val metric = eval.evaluate(model.transform(validationDataset, paramMap))
          instr.logDebug(s"Got metric $metric for model trained with $paramMap.")
          metric
        } (executionContext)
      }

      // Wait for metrics to be calculated
      val foldMetrics = foldMetricFutures.map(ThreadUtils.awaitResult(_, Duration.Inf))

      // Unpersist training & validation set once all metrics have been produced
      trainingDataset.unpersist()
      validationDataset.unpersist()
      foldMetrics
    }.transpose

    // The metrics now is 2-D array (cross validation results per parameter configuration)
    metrics.zipWithIndex foreach {
      case (cvrs, idx) =>
        println(s"Cross validation results for parameter configuration ${epm(idx)}")
        cvrs.zipWithIndex foreach { // Print cross validation results for configuration
          case (cv, idx2) =>
            println(s"(${idx2}, ${cv})")
        }
    }

    val avg = metrics.map(_.sum / $(numFolds))
    instr.logInfo(s"Average cross-validation metrics: ${avg.toSeq}")
    val (bestMetric, bestIndex) =
      if (eval.isLargerBetter) avg.zipWithIndex.maxBy(_._1)
      else avg.zipWithIndex.minBy(_._1)
    instr.logInfo(s"Best set of parameters:\n${epm(bestIndex)}")
    instr.logInfo(s"Best cross-validation metric: $bestMetric.")
    val bestModel = est.fit(dataset, epm(bestIndex)).asInstanceOf[Model[_]]
    copyValues(new CrossValidatorModel(uid, bestModel, avg)
      .setSubModels(subModels).setParent(this))
  }
}

