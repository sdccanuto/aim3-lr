package com.celebihacker.ml.logreg;

import org.apache.hadoop.util.ToolRunner;
import org.junit.Test;

import com.celebihacker.ml.logreg.ensemble.EnsembleJob;
import com.celebihacker.ml.logreg.ensemble.EvalJob;

public class EnsembleJobTest {

  @Test
  public void test() throws Exception {
//    String[] args = new String[] { inputPath, outputPath };
    ToolRunner.run(new EnsembleJob(), null);
    
    ToolRunner.run(new EvalJob(), null);
  }

}
