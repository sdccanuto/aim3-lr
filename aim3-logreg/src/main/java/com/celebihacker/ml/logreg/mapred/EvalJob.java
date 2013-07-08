package com.celebihacker.ml.logreg.mapred;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.mahout.common.IntPairWritable;

public class EvalJob extends AbstractHadoopJob {

  private static String JOB_NAME = "aim3-validation";

  private static final int REDUCE_TASKS = 1;
  
  private String inputFile;
  private String outputPath;
  
  public EvalJob(String inputFile,
      String outputPath) {
    
    this.inputFile = inputFile;
    this.outputPath = outputPath;
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
    
    return job.waitForCompletion(true) ? 0 : 1;
  }

}
