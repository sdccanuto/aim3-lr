package com.celebihacker.ml.logreg.ensemble;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.celebihacker.ml.AdaptiveLogger;
import com.celebihacker.ml.VectorLabeledWritable;

public class EnsembleReducer extends Reducer<IntWritable, VectorLabeledWritable, IntWritable, VectorWritable> {
  
  Text curOutput = new Text();
  
  private static AdaptiveLogger log = new AdaptiveLogger(
      EnsembleJob.RUN_LOCAL_MODE, Logger.getLogger(EnsembleReducer.class.getName())); 
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    log.setLevel(Level.DEBUG);
  }

  @Override
  public void reduce(IntWritable key, Iterable<VectorLabeledWritable> values, Context context) throws IOException, InterruptedException {
    // TRAIN
    // Use stochastic gradient descent online learning
    OnlineLogisticRegression learningAlgorithm = new OnlineLogisticRegression(
    EnsembleJob.TARGETS, EnsembleJob.FEATURES, new L1());
    learningAlgorithm.alpha(1).stepOffset(1000)
    .decayExponent(0.2)
    .lambda(3.0e-5)
    .learningRate(20);

    int total = 0;
    int correct = 0;
    for (VectorLabeledWritable lVec : values) {
      
      // Test prediction
      int actualTarget = lVec.getLabel();
      Vector vec = lVec.getVector();
      double prediction = learningAlgorithm.classifyScalar(vec);
      if (actualTarget == Math.round(prediction)) ++correct;

      learningAlgorithm.train(actualTarget, vec);
      
      ++total;
    }
    log.debug("ONLINE TRAINING RESULTS:");
    log.debug("Total: " + total);
    log.debug("Correct: " + correct);
    log.debug("Accuracy: " + ((double)correct)/((double)total));
    learningAlgorithm.close();
    
    context.write(key, new VectorWritable(learningAlgorithm.getBeta().viewRow(0)));
  }

}
