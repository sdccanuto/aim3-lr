package com.celebihacker.ml.logreg.mapred;

import java.util.Arrays;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;

import edu.stanford.nlp.optimization.DiffFunction;
import edu.stanford.nlp.optimization.QNMinimizer;

/**
 * Batch gradient for logistic regression
 */
public class LBFGSJob extends Configured implements Tool {

  private GradientJob gradientJob;
  private TrainingErrorJob trainingErrorJob;
  private double eps;
  private final int maxIterations;
  private double[] initial;

  public LBFGSJob(TrainingErrorJob trainingErrorJob, GradientJob gradientJob, 
      double eps, int maxIterations, double[] initial) {

    this.trainingErrorJob = trainingErrorJob;
    this.gradientJob = gradientJob;
    this.eps = eps;
    this.maxIterations = maxIterations;
    this.initial = initial;
  }

  @Override
  public int run(String[] args) throws Exception {
    
    DiffFunction f = new LBFGSDiffFunction(this.trainingErrorJob, this.gradientJob);
    
    QNMinimizer qn = new QNMinimizer(15, true);
    double[] model = qn.minimize(f, this.eps, this.initial, this.maxIterations);
    
    System.out.println(Arrays.toString(model));
    
    return 0;
  }
}