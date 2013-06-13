package com.celebihacker.ml.logreg;

import java.io.IOException;

import org.junit.Test;

import com.celebihacker.ml.logreg.LogRegRCV1Local;

public class RCV1SeqLogRegTest {

  @Test
  public void testTrainRCV1() throws IOException {
    String trainingFile = "/home/andre/dev/datasets/RCV1-v2/vectors/lyrl2004_vectors_train.dat";
    String testFile = "/home/andre/dev/datasets/RCV1-v2/vectors/lyrl2004_vectors_test_pt0.dat";
    
    LogRegRCV1Local lr = new LogRegRCV1Local();
    lr.trainRCV1(trainingFile, testFile);
  }

}
