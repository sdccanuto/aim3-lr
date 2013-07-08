package com.celebihacker.ml.logreg;

import org.apache.hadoop.util.ToolRunner;
import org.junit.Test;

import com.celebihacker.ml.logreg.mapred.BatchGradientJob;

public class BatchGradientJobTest {

  public static final boolean RUN_LOCAL_MODE = true;
  
//  static final String INPUT_FILE_TRAIN_LOCAL = "/Users/uce/Desktop/rcv1-v2/vectors/lyrl2004_vectors_train_5000.seq";
//  static final String INPUT_FILE_TRAIN_HDFS ="hdfs://localhost:9000/user/uce/rcv1/rcv1-v2/lyrl2004_vectors_train_5000.seq";
  static final String INPUT_FILE_TRAIN_LOCAL = "/home/andre/dev/datasets/RCV1-v2/vectors/lyrl2004_vectors_train_5000.seq";
  static final String INPUT_FILE_TRAIN_HDFS ="datasets/rcv1-v2/lyrl2004_vectors_train_5000.seq";
  
  static final String OUTPUT_TRAIN_PATH = "output-aim3-batchgd";
  
  static final int MAX_ITERATIONS = 3;

  @Test
  public void test() throws Exception {

    double initial = 1;
    
    BatchGradientJob job = new BatchGradientJob(
        INPUT_FILE_TRAIN_LOCAL, 
        OUTPUT_TRAIN_PATH, 
        MAX_ITERATIONS,
        initial);
    
//    job.setWeightVector(initial);
    
    ToolRunner.run(job, null);
  }
}