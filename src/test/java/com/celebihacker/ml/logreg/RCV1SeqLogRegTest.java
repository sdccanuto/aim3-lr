package com.celebihacker.ml.logreg;

import java.io.IOException;

import org.junit.Test;

import com.celebihacker.ml.logreg.LogRegRCV1;

public class RCV1SeqLogRegTest {

  @Test
  public void testTrainRCV1() throws IOException {
    LogRegRCV1 lr = new LogRegRCV1();
    lr.trainRCV1();
  }

}
