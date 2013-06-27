package com.celebihacker.ml.logreg.mapred;

import java.io.IOException;

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

import com.celebihacker.ml.logreg.EnsembleJobTest;
import com.celebihacker.ml.util.AdaptiveLogger;

public class GradientReducer extends Reducer<NullWritable, VectorWritable, NullWritable, VectorWritable> {
  
  private static AdaptiveLogger log = new AdaptiveLogger(
      EnsembleJobTest.RUN_LOCAL_MODE, Logger.getLogger(GradientReducer.class.getName()), Level.DEBUG); 
  
  @Override
  public void reduce(NullWritable key, Iterable<VectorWritable> values, Context context) throws IOException, InterruptedException {
    Vector batchGradientSum = new RandomAccessSparseVector((int)EnsembleJob.datasetInfo.getVectorSize());
    
    for (VectorWritable gradient : values) {
      batchGradientSum.assign(gradient.get(), Functions.PLUS);
    }
    log.debug("Gradient result: Dimensions: " + batchGradientSum.size() + " Non Zero: " + batchGradientSum.getNumNonZeroElements());
  }

}
