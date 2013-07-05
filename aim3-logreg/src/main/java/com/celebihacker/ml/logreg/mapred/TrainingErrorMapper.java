package com.celebihacker.ml.logreg.mapred;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.celebihacker.ml.logreg.GradientJobTest;
import com.celebihacker.ml.util.AdaptiveLogger;
import com.celebihacker.ml.writables.IDAndLabels;

public class TrainingErrorMapper extends
    Mapper<IDAndLabels, VectorWritable, NullWritable, DoubleWritable> {

  static final int LABEL_DIMENSION = EnsembleJob.datasetInfo
      .getLabelIdByName(EnsembleJob.TARGET_POSITIVE);

  private static AdaptiveLogger LOGGER = new AdaptiveLogger(
      GradientJobTest.RUN_LOCAL_MODE, Logger.getLogger(TrainingErrorMapper.class.getName()),
      Level.DEBUG);

  private Vector weights;
  private DoubleWritable trainingError = new DoubleWritable();

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);

    // read initial weights
    Configuration conf = context.getConfiguration();
    Path[] iterationWeights = DistributedCache.getLocalCacheFiles(conf);

    if (iterationWeights == null) { throw new RuntimeException("No weights set"); }

    Path localPath = new Path("file://" + iterationWeights[0].toString());

    for (Pair<NullWritable, VectorWritable> weights : new SequenceFileIterable<NullWritable, VectorWritable>(
        localPath, conf)) {

      this.weights = weights.getSecond().get();
    }
  }

  @Override
  public void map(IDAndLabels key, VectorWritable value, Context context) throws IOException,
      InterruptedException {

    Vector x = value.get();
    double y = key.getLabels().get(LABEL_DIMENSION);

    double diff = predict(x, this.weights) - y;
    this.trainingError.set(Math.pow(diff, 2));
    
    context.write(NullWritable.get(), this.trainingError);
  }

  public double predict(Vector x, Vector w) {
    return 1.0 / (1.0 + Math.exp(-x.dot(w)));
  }
}