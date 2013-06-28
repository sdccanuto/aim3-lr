package com.celebihacker.ml.logreg;

import java.util.List;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;

import com.celebihacker.ml.ClassificationModel;
import com.celebihacker.ml.RegressionModel;
import com.celebihacker.ml.util.MLUtils;

/**
 * Implementation of a ensemble model for Logistic Regression
 * Internally it has multiple logistic regression models
 * Those are combined to a single prediction (supporting different VotingSchemas)
 * 
 */
public class LogisticRegressionEnsemble implements ClassificationModel, RegressionModel {
  
  List<RandomAccessSparseVector> models;
  double threshold;
  VotingSchema votingSchema;
  Vector mergedModel;   // only used if votingSchema is merge_model
  
  LogisticRegression logreg;
  
  public enum VotingSchema {
    MAJORITY_VOTE,
    MERGED_MODEL
  }

  public LogisticRegressionEnsemble(List<RandomAccessSparseVector> models, double threshold, VotingSchema votingSchema) {
    this.models = models;
    this.threshold = threshold;
    this.votingSchema = votingSchema;
    this.logreg = new LogisticRegression(null, threshold);
    
    if (votingSchema == VotingSchema.MERGED_MODEL) {
      // TODO Interesting: How much vary the features between the models? How much sparsity is removed on merging?
      
      RandomAccessSparseVector[] modelArray = models.toArray(new RandomAccessSparseVector[models.size()]);
      Matrix modelsMatrix = new SparseRowMatrix(models.size(), models.get(0).size(), modelArray);
      mergedModel = MLUtils.meanByColumns(modelsMatrix);
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
      
      int[] votes = new int[2];
      for (Vector w : models) {
        logreg.setW(w);
        ++ votes[ logreg.classify(x) ];
      }
      // TODO Show how the variance is between the different models
      // TODO Show warning if number is even (no majority might exist)
      prediction = votes[1] > votes[0] ? 1 : 0;
      
      break;
      
    case MERGED_MODEL:

      double xDotW = x.dot(mergedModel) + intercept;
      prediction = LogisticRegression.logisticFunction(xDotW);
      
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
