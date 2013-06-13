package com.celebihacker.ml.logreg;

import org.apache.hadoop.util.ToolRunner;
import org.junit.Test;

public class RCV1JobTest {

  @Test
  public void test() throws Exception {
//    String[] args = new String[] { inputPath, outputPath };
    ToolRunner.run(new RCV1Job(), null);
    System.out.println("Done");
  }

}
