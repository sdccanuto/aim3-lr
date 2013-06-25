package com.celebihacker.ml.logreg.ensemble;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.celebihacker.ml.VectorLabeledWritable;
import com.celebihacker.ml.util.AdaptiveLogger;
import com.celebihacker.ml.validation.OnlineAccuracy;

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

    OnlineAccuracy accuracy = new OnlineAccuracy(0.5);
    for (VectorLabeledWritable lVec : values) {
      
      // Test prediction
      int actualTarget = lVec.getLabel();
      Vector vec = lVec.getVector();
      double prediction = learningAlgorithm.classifyScalar(vec);
      accuracy.addSample(actualTarget, prediction);

      learningAlgorithm.train(actualTarget, vec);
    }
    log.debug("ONLINE TRAINING RESULTS:");
    log.debug("Accuracy: " + accuracy.getAccuracy() + " (= " + (accuracy.getTrueNegatives() + accuracy.getTruePositives()) + " / " + accuracy.getTotal() + ")");
    learningAlgorithm.close();
    
    RandomAccessSparseVector w = new RandomAccessSparseVector(learningAlgorithm.getBeta().viewRow(0));
    context.write(key, new VectorWritable(w));
  }

}
