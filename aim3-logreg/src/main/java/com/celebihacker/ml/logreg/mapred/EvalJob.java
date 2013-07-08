package com.celebihacker.ml.logreg.mapred;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.mahout.common.IntPairWritable;

public class EvalJob extends AbstractHadoopJob {

  private static String JOB_NAME = "aim3-validation";
  
  static String PARAM_NAME_TRAIN_OUTPUT = "train-output";

  private static final int REDUCE_TASKS = 1;
  
  private String inputFile;
  private String outputPath;
  private String trainOuputPath;
  
  public EvalJob(String inputFile,
      String outputPath,
      String trainOuputPath) {
    
    this.inputFile = inputFile;
    this.outputPath = outputPath;
    this.trainOuputPath = trainOuputPath;
  }

  public int run(String[] args) throws Exception {
    
    Job job = prepareJob(
        JOB_NAME, 
        REDUCE_TASKS, 
        EvalMapper.class, 
        EvalReducer.class, 
        Text.class,
        IntPairWritable.class,
        Text.class,
        Text.class,
        SequenceFileInputFormat.class,
        TextOutputFormat.class,
        inputFile,
        outputPath);
    
    cleanupOutputDirectory(outputPath);
    
    job.getConfiguration().set(PARAM_NAME_TRAIN_OUTPUT, trainOuputPath);
    
    return job.waitForCompletion(true) ? 0 : 1;
  }

}
