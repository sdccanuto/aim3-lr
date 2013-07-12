package com.celebihacker.ml.logreg;

import org.junit.Test;

import com.celebihacker.ml.datasets.DatasetInfo;
import com.celebihacker.ml.datasets.RCV1DatasetInfo;
import com.celebihacker.ml.logreg.iterative.BatchGradientDriver;

public class BatchGradientJobTest {
  
  private static final DatasetInfo DATASET = RCV1DatasetInfo.get();
  private static final String TARGET_POSITIVE = "CCAT";

//  private static final String INPUT_FILE_TRAIN_LOCAL = "/Users/uce/Desktop/rcv1-v2/vectors/lyrl2004_vectors_train_5000.seq";
//  private static final String INPUT_FILE_TRAIN_HDFS ="hdfs://localhost:9000/user/uce/rcv1/rcv1-v2/lyrl2004_vectors_train_5000.seq";
  private static final String INPUT_FILE_TRAIN_LOCAL = "/home/andre/dev/datasets/RCV1-v2/vectors/lyrl2004_vectors_train_5000.seq";
//  private static final String INPUT_FILE_TRAIN_HDFS ="datasets/rcv1-v2/lyrl2004_vectors_train_5000.seq";
  
  private static final String OUTPUT_TRAIN_PATH = "output-aim3-batchgd";
  
  private static final int MAX_ITERATIONS = 3;

  @Test
  public void test() throws Exception {
    
    int labelDimension = DATASET.getLabelIdByName(TARGET_POSITIVE);

    double initial = 1;
    
    BatchGradientDriver bgDriver = new BatchGradientDriver(
        INPUT_FILE_TRAIN_LOCAL, 
        OUTPUT_TRAIN_PATH, 
        MAX_ITERATIONS,
        initial,
        labelDimension,
        (int)DATASET.getVectorSize());
    
//    job.setWeightVector(initial);
    
    bgDriver.train();
  }
}