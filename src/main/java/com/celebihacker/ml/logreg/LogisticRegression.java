package com.celebihacker.ml.logreg;

import org.apache.mahout.math.Vector;

import com.celebihacker.ml.ClassificationModel;
import com.celebihacker.ml.RegressionModel;


/**
 * Functions related to Logistic Regression, mostly for training and evaluation
 * Can be used as instance of a concrete model, but also offers public static methods.
 * 
 * TODO Make a separate class just with static methods
 * 
 * @author andre
 */
public class LogisticRegression implements RegressionModel, ClassificationModel {
  
  private Vector w;
  private double threshold;
  
  public LogisticRegression(Vector w, double threshold) {
    this.w = w;
    this.threshold = threshold;
  }
  
  @Override
  public double predict(Vector x) {
    return predict(x, 0);
  }

  @Override
  public double predict(Vector x, double intercept) {
    double xDotW = x.dot(w) + intercept;
    return logisticFunction(xDotW);
  }

  public static double logisticFunction(double exponent) {
    double negativeExp = Math.exp(-exponent);
    if (exponent != 0 && (negativeExp == 0 || Double.isInfinite(negativeExp))) {
      System.out.println(" - OVERFLOW? " + exponent + "\t" + negativeExp);
    }
    return 1d / (1d + negativeExp);
  }

  @Override
  public int classify(Vector x) {
    return (int)Math.floor(predict(x) + threshold);
  }

  /**
   * Compute the partial gradient of negative log-likelihood function regarding a single data point x
   * @return ( h(x) - y) * x
   */
  public Vector computePartialGradient(Vector x, double y) {
    double diff = predict(x) - y;
    
    return x.times(diff);
  }

  
  public void setW(Vector w) {
    this.w = w;
  }
  
  public void setThreshold(double threshold) {
    this.threshold = threshold;
  }
  
  public double getThreshold() {
    return threshold;
  }

}
