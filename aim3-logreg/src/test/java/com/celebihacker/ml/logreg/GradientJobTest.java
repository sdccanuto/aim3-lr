package com.celebihacker.ml.logreg;

import org.apache.hadoop.util.ToolRunner;
import org.junit.Test;

import com.celebihacker.ml.logreg.mapred.GradientJob;

public class GradientJobTest {

  public static final boolean RUN_LOCAL_MODE = true;
  
  static final String INPUT_FILE_TRAIN_LOCAL = "/home/andre/dev/datasets/RCV1-v2/vectors/lyrl2004_vectors_train.seq";
  static final String INPUT_FILE_TRAIN_HDFS = "rcv1-v2/lyrl2004_vectors_train.seq";
  
  static final String OUTPUT_TRAIN_PATH = "output-aim3-ensemble";
  
  static final String JAR_PATH = "target/aim3-logreg-0.0.1-SNAPSHOT-job.jar";
  static final String CONFIG_FILE_PATH = "core-site.xml";

  @Test
  public void test() throws Exception {
    ToolRunner.run(new GradientJob(
        INPUT_FILE_TRAIN_LOCAL,
        INPUT_FILE_TRAIN_HDFS, 
        OUTPUT_TRAIN_PATH, 
        JAR_PATH, 
        CONFIG_FILE_PATH,
        RUN_LOCAL_MODE), null);
  }
}
