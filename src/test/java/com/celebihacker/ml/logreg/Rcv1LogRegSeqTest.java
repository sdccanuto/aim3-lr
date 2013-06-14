package com.celebihacker.ml.logreg;

import java.io.IOException;

import org.junit.Test;

import com.celebihacker.ml.logreg.sequential.Rcv1LogRegSeq;

public class Rcv1LogRegSeqTest {

  @Test
  public void testTrainRCV1() throws IOException {
    String trainingFile = "/home/andre/dev/datasets/RCV1-v2/vectors/lyrl2004_vectors_train_5000.dat";
    String testFile = "/home/andre/dev/datasets/RCV1-v2/vectors/lyrl2004_vectors_test_pt0.dat";
    
    Rcv1LogRegSeq lr = new Rcv1LogRegSeq();
    lr.trainRCV1(trainingFile, testFile);
  }

}
