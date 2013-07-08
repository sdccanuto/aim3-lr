package com.celebihacker.ml.logreg;

import org.apache.hadoop.util.ToolRunner;
import org.junit.Test;

import com.celebihacker.ml.logreg.mapred.EnsembleJob;
import com.celebihacker.ml.logreg.mapred.EvalJob;

public class EnsembleJobTest {
  
  public static final int ENSEMBLE_SIZE = 4;
  
  static final String INPUT_FILE_TRAIN_LOCAL = "/home/andre/dev/datasets/RCV1-v2/vectors/lyrl2004_vectors_train_10000.seq";
  static final String INPUT_FILE_TRAIN_HDFS = "rcv1-v2/lyrl2004_vectors_train_5000.seq";
  
  private static final String INPUT_FILE_TEST_LOCAL = "/home/andre/dev/datasets/RCV1-v2/vectors/lyrl2004_vectors_test_5000.seq";
//  private static final String INPUT_FILE_TEST_HDFS = "rcv1-v2/lyrl2004_vectors_test_5000.seq";
  
  // TODO Send this path via Distributed Cache to EvalMapper
  public static final String OUTPUT_TRAIN_PATH = "output-aim3-ensemble";
  private static final String OUTPUT_TEST_PATH = "output-aim3-validation";

  @Test
  public void test() throws Exception {
    
//    String[] args = new String[] { inputPath, outputPath };
    ToolRunner.run(new EnsembleJob(
        INPUT_FILE_TRAIN_LOCAL, 
        OUTPUT_TRAIN_PATH, 
        ENSEMBLE_SIZE), null);
    
    ToolRunner.run(new EvalJob(
        INPUT_FILE_TEST_LOCAL, 
        OUTPUT_TEST_PATH), null);
  }

}
