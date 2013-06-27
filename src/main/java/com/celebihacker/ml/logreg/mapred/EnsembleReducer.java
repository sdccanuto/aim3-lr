package com.celebihacker.ml.logreg.mapred;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.celebihacker.ml.VectorMultiLabeledWritable;
import com.celebihacker.ml.logreg.EnsembleJobTest;
import com.celebihacker.ml.util.AdaptiveLogger;
import com.celebihacker.ml.validation.OnlineAccuracy;

public class EnsembleReducer extends Reducer<IntWritable, VectorMultiLabeledWritable, IntWritable, VectorWritable> {
  
  static final int LABEL_DIMENSION = EnsembleJob.datasetInfo.getLabelIdByName(EnsembleJob.TARGET_POSITIVE);
  
  private static AdaptiveLogger log = new AdaptiveLogger(
      EnsembleJobTest.RUN_LOCAL_MODE, Logger.getLogger(EnsembleReducer.class.getName()), Level.DEBUG); 

  @Override
  public void reduce(IntWritable key, Iterable<VectorMultiLabeledWritable> values, Context context) throws IOException, InterruptedException {
    // TRAIN
    // Use stochastic gradient descent online learning
    OnlineLogisticRegression learningAlgorithm = new OnlineLogisticRegression(
    2, (int)EnsembleJob.datasetInfo.getVectorSize(), new L1());
    learningAlgorithm.alpha(1).stepOffset(1000)
    .decayExponent(0.2)
    .lambda(3.0e-5)
    .learningRate(20);

    OnlineAccuracy accuracy = new OnlineAccuracy(0.5);
    for (VectorMultiLabeledWritable lVec : values) {
      
      // Test prediction
      int actualTarget = (int)lVec.getLabels().get(LABEL_DIMENSION);
      Vector vec = lVec.getVector();
      double prediction = learningAlgorithm.classifyScalar(vec);
      accuracy.addSample(actualTarget, prediction);

      // Train
      learningAlgorithm.train(actualTarget, vec);
    }
    log.debug("ONLINE TRAINING RESULTS:");
    log.debug("Accuracy: " + accuracy.getAccuracy() + " (= " + (accuracy.getTrueNegatives() + accuracy.getTruePositives()) + " / " + accuracy.getTotal() + ")");
    learningAlgorithm.close();
    
    RandomAccessSparseVector w = new RandomAccessSparseVector(learningAlgorithm.getBeta().viewRow(0));
    context.write(key, new VectorWritable(w));
  }

}
