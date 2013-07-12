package com.celebihacker.ml.logreg.ensemble;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.math.VectorWritable;

import com.celebihacker.ml.AbstractHadoopJob;
import com.celebihacker.ml.writables.VectorMultiLabeledWritable;

public class EnsembleJob extends AbstractHadoopJob {

  private static String JOB_NAME = "aim3-ensemble-train";
  
  private String inputFile;
  private String outputPath;
  private int partitions;
  private int labelDimension;
  
  static final String CONF_KEY_LABEL_DIMENSION = "label-dimension";
  
  public EnsembleJob(
      String inputFile,
      String outputPath,
      int partitions,
      int labelDimension) {
    this.inputFile = inputFile;
    this.outputPath = outputPath;
    this.partitions = partitions;
    this.labelDimension = labelDimension;
  }

  public int run(String[] args) throws Exception {

    Job job = prepareJob(
        JOB_NAME, 
        partitions, 
        EnsembleMapper.class, 
        EnsembleReducer.class, 
        IntWritable.class,
        VectorMultiLabeledWritable.class,
        IntWritable.class,
        VectorWritable.class,
        SequenceFileInputFormat.class,
        SequenceFileOutputFormat.class,
        inputFile,
        outputPath);
    
    job.getConfiguration().set(CONF_KEY_LABEL_DIMENSION, Integer.toString(labelDimension));
    
    cleanupOutputDirectory(outputPath);
    
    return job.waitForCompletion(true) ? 0 : 1;
  }

}
