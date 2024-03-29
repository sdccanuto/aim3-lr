package com.celebihacker.ml.logreg.ensemble;

import java.util.List;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;

import com.celebihacker.ml.ClassificationModel;
import com.celebihacker.ml.RegressionModel;
import com.celebihacker.ml.logreg.LogRegMath;
import com.celebihacker.ml.util.MLUtils;

/**
 * Implementation of a ensemble model for Logistic Regression
 * Internally it has multiple logistic regression models
 * Those are combined to a single prediction (supporting different VotingSchemas)
 */
public class LogRegEnsembleModel implements ClassificationModel, RegressionModel {
  
  List<RandomAccessSparseVector> models;
  double threshold;
  VotingSchema votingSchema;
  Vector wMergedModel;   // only used if votingSchema is merge_model
  
  public enum VotingSchema {
    MAJORITY_VOTE,
    MERGED_MODEL
  }

  public LogRegEnsembleModel(List<RandomAccessSparseVector> models, double threshold, VotingSchema votingSchema) {
    this.models = models;
    this.threshold = threshold;
    this.votingSchema = votingSchema;
    
    if (votingSchema == VotingSchema.MERGED_MODEL) {
      // TODO Interesting: How much vary the features between the models? How much sparsity is removed on merging?
      
      RandomAccessSparseVector[] modelArray = models.toArray(new RandomAccessSparseVector[models.size()]);
      Matrix modelsMatrix = new SparseRowMatrix(models.size(), models.get(0).size(), modelArray);
      wMergedModel = MLUtils.meanByColumns(modelsMatrix);
    }
  }

  @Override
  public double predict(Vector x) {
    return predict(x, 0);
  }

  @Override
  public double predict(Vector x, double intercept) {
    
    double prediction = -1;
    
    switch (votingSchema) {
    case MAJORITY_VOTE:
      
      double[] votes = new double[2];
      for (Vector w : models) {
        votes[ LogRegMath.classify(x, w) ] += LogRegMath.predict(x, w); 
      }
      // TODO Show how the variance is between the different models
      // TODO Show warning if number is even (no majority might exist)
      prediction = votes[1] > votes[0] ? 1 : 0;
      
      break;
      
    case MERGED_MODEL:

      prediction = LogRegMath.predict(x, wMergedModel, intercept);
      
      break;
      
    default:
      break;
    }
    
    return prediction;
  }

  @Override
  public int classify(Vector x) {
    return (int)Math.floor(predict(x) + threshold);
  }

}
