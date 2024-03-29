package com.celebihacker.ml.logreg.iterative;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.celebihacker.ml.AbstractHadoopJob;
import com.celebihacker.ml.datasets.RCV1DatasetInfo;
import com.celebihacker.ml.util.HadoopUtils;

/**
 * Computes the in-sample error for logistic regression
 */
public class TrainingErrorJob extends AbstractHadoopJob {

  private static String JOB_NAME = "aim3-training-error";

  static final int REDUCE_TASKS = 1;

  private String inputFile;
  private String outputPath;
  private int labelDimension;
  
  static final String CONF_KEY_LABEL_DIMENSION = "label-dimension";

  private final VectorWritable w;

  public TrainingErrorJob(
      String inputFile,
      String outputPath,
      int labelDimension) {
    this.inputFile = inputFile;
    this.outputPath = outputPath;
    this.labelDimension = labelDimension;

    Vector weights = new SequentialAccessSparseVector((int) RCV1DatasetInfo.get().getNumFeatures());
    this.w = new VectorWritable(weights);
  }

  @Override
  public int run(String[] args) throws Exception {
    
    Job job = prepareJob(
        JOB_NAME, 
        REDUCE_TASKS, 
        TrainingErrorMapper.class, 
        TrainingErrorReducer.class, 
        NullWritable.class,
        DoubleWritable.class,
        NullWritable.class,
        DoubleWritable.class,
        SequenceFileInputFormat.class,
        SequenceFileOutputFormat.class,
        inputFile,
        outputPath);
    job.setCombinerClass(TrainingErrorReducer.class);
    
    job.getConfiguration().set(CONF_KEY_LABEL_DIMENSION, Integer.toString(labelDimension));
    
    cleanupOutputDirectory(outputPath);

    // Initial weights
    Path cachePath = new Path(job.getConfiguration().get("hadoop.tmp.dir") + "/initial_weights");
    HadoopUtils.writeVectorToDistCache(job.getConfiguration(), this.w, cachePath);

    return job.waitForCompletion(true) ? 0 : 1;
  }

  public void setWeightVector(double initial) {
    this.w.get().assign(initial);
  }

  public void setWeightVector(double[] weights) {
    this.w.get().assign(weights);
  }

  public void setWeightVector(Vector weights) {
    this.w.get().assign(weights);
  }
  
  public String getOutputPath() {
    return this.outputPath;
  }
}