package com.celebihacker.ml.logreg;

import org.apache.hadoop.util.ToolRunner;
import org.junit.Test;

import com.celebihacker.ml.datasets.RCV1DatasetInfo;
import com.celebihacker.ml.logreg.iterative.TrainingErrorJob;

public class TrainingErrorJobTest {

  private static final String TARGET_POSITIVE = "CCAT";

  private static final String INPUT_FILE_TRAIN_LOCAL = "/Users/uce/Desktop/rcv1-v2/vectors/lyrl2004_vectors_train_5000.seq";
//  private static final String INPUT_FILE_TRAIN_HDFS ="hdfs://localhost:9000/user/uce/rcv1/rcv1-v2/lyrl2004_vectors_train_5000.seq";
  private static final String OUTPUT_TRAIN_PATH = "output-aim3-gradient";

  @Test
  public void test() throws Exception {
    
    int labelDimension = RCV1DatasetInfo.get().getLabelIdByName(TARGET_POSITIVE);
    
    TrainingErrorJob job = new TrainingErrorJob(
        INPUT_FILE_TRAIN_LOCAL,
        OUTPUT_TRAIN_PATH,
        labelDimension);
    
    ToolRunner.run(job, null);
  }
}