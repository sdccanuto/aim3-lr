package com.celebihacker.ml.logreg.mapred;

import java.io.IOException;

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

import com.celebihacker.ml.logreg.GradientJobTest;
import com.celebihacker.ml.logreg.LogisticRegression;
import com.celebihacker.ml.util.AdaptiveLogger;
import com.celebihacker.ml.writables.IDAndLabels;

public class GradientMapper extends Mapper<IDAndLabels, VectorWritable, NullWritable, VectorWritable> {
  
  static final int LABEL_DIMENSION = EnsembleJob.datasetInfo.getLabelIdByName(EnsembleJob.TARGET_POSITIVE);
  
  private static AdaptiveLogger log = new AdaptiveLogger(
      GradientJobTest.RUN_LOCAL_MODE, Logger.getLogger(GradientMapper.class.getName()), Level.DEBUG); 
  
  private LogisticRegression logreg;
  
  private Vector batchGradient = new RandomAccessSparseVector((int)EnsembleJob.datasetInfo.getVectorSize());
  private long count=0;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    log.debug("Map Setup");
    
    // TODO Read model from distributed cache (or from hdfs)
    Vector w = new RandomAccessSparseVector((int)EnsembleJob.datasetInfo.getVectorSize());
    logreg = new LogisticRegression(w, 0.5d);
  }

  @Override
  public void map(IDAndLabels key, VectorWritable value, Context context) throws IOException, InterruptedException {
    
    // Compute gradient regarding current data point
    Vector gradient = logreg.computePartialGradient(value.get(), (int)key.getLabels().get(LABEL_DIMENSION));
    batchGradient.assign(gradient, Functions.PLUS);
    ++count;
  }
  
  @Override
  protected void cleanup(Context context)
      throws IOException, InterruptedException {
    super.cleanup(context);
    context.write(NullWritable.get(), new VectorWritable(batchGradient));
    log.debug("Mapper: partial sum of gradient (" + count + " items)");
    log.debug("- Dimensions: " + batchGradient.size() + " Non Zero: " + batchGradient.getNumNonZeroElements());
  }
  
}