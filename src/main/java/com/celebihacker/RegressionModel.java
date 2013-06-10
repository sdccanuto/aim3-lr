package com.celebihacker;

import org.apache.mahout.math.Vector;

public interface RegressionModel {
  
  double predict(Vector x, Vector w);

}
