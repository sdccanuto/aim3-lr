package com.celebihacker;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

/**
 * @author andre
 */
public class Validation {  
  
  Matrix confusion;
  double accuracy;
  double meanDeviation;  
  
  public void computeMeanDeviation(Matrix data, Vector y, Vector w, RegressionModel model) {
    double dev = 0;
    for (int n=0; n<data.numRows(); ++n) {
      //System.out.println(data.viewRow(n));
      double prediction = model.predict(data.viewRow(n), w);
      dev += Math.abs((y.get(n) - prediction));
    
    //    System.out.println("Is: " + prediction + " should: " + y.get(n));
    }
    this.meanDeviation = dev / data.numRows(); 
  }

  public void computeAccuracy(Matrix data, Vector y, Vector w, ClassificationModel model) {
    // How many do we classify correctly?
    confusion = new DenseMatrix(2,2);
    int truePos = 0;
    int trueNeg = 0;
    int falsePos = 0;
    int falseNeg = 0;
    for (int n=0; n<data.numRows(); ++n) {
      //System.out.println(data.viewRow(n));
      double prediction = model.classify(data.viewRow(n), w);
  //    System.out.println("Is: " + prediction + " should: " + y.get(n));
      if (Math.round(prediction) == y.get(n)) {
        if (y.get(n) == 0)
          ++truePos;
        else
          ++trueNeg;
      } else {
        if (y.get(n) == 0)
          ++falseNeg;
        else
          ++falsePos;
      }
    }
    confusion.set(0, 0, truePos);
    confusion.set(0, 1, falseNeg);
    confusion.set(1, 0, falsePos);
    confusion.set(1, 1, trueNeg);
    this.accuracy = ((double)truePos + trueNeg) / ((double)data.numRows()); 
  }
  
  public Matrix getConfusion() {
    return confusion;
  }
  
  public double getMeanDeviation() {
    return meanDeviation;
  }
  
  public double getAccuracy() {
    return accuracy;
  }

}
