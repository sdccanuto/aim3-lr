package com.celebihacker.ml.logreg.mapred;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.math.VectorWritable;

import com.celebihacker.ml.datasets.DatasetInfo;
import com.celebihacker.ml.datasets.RCV1DatasetInfo;
import com.celebihacker.ml.writables.VectorMultiLabeledWritable;

public class EnsembleJob extends AbstractHadoopJob {

  private static String JOB_NAME = "aim3-ensemble-train";
  
  private String inputFile;
  private String outputPath;
  private int partitions;
  
  static DatasetInfo datasetInfo = RCV1DatasetInfo.get();
  
  // TODO Feature: Currently we train a hardcoded single 1-vs-all classifier
  static final String TARGET_POSITIVE = "CCAT";
  
  public EnsembleJob(
      String inputFile,
      String outputPath,
      int partitions) {
    this.inputFile = inputFile;
    this.outputPath = outputPath;
    this.partitions = partitions;
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
    
    cleanupOutputDirectory(outputPath);
    
    return job.waitForCompletion(true) ? 0 : 1;
  }

}
