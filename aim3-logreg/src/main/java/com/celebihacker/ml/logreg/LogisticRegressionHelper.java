package com.celebihacker.ml.logreg;

import org.apache.mahout.math.Vector;

/**
 * Static methods needed for logistic regression in MapReduce jobs.
 * 
 * TODO this should be merged with LogisticRegression (why do we need to create an instance of
 * LogisticRegression?)
 */
public class LogisticRegressionHelper {

  public static double predict(Vector x, Vector w) {
    return logisticFunction(x.dot(w));
  }

  public static Vector computePartialGradient(Vector x, Vector w, double y) {
    return x.times(predict(x, w) - y);
  }

  public static double computeError(Vector x, Vector w, double y) {
    return Math.pow(predict(x, w) - y, 2);
  }

  public static double logisticFunction(double exp) {
    return 1d / (1d + Math.exp(-exp));
  }
}