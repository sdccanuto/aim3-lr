package com.celebihacker.ml.logreg.mapred;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.celebihacker.ml.logreg.LogisticRegressionHelper;
import com.celebihacker.ml.writables.IDAndLabels;

public class TrainingErrorMapper extends
    Mapper<IDAndLabels, VectorWritable, NullWritable, DoubleWritable> {

  static final int LABEL_DIMENSION = EnsembleJob.datasetInfo
      .getLabelIdByName(EnsembleJob.TARGET_POSITIVE);

  private Vector w;
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

      this.w = weights.getSecond().get();
    }
  }

  @Override
  public void map(IDAndLabels key, VectorWritable value, Context context) throws IOException,
      InterruptedException {

    Vector x = value.get();
    double y = key.getLabels().get(LABEL_DIMENSION);

    this.trainingError.set(LogisticRegressionHelper.computeError(x, this.w, y));

    context.write(NullWritable.get(), this.trainingError);
  }
}