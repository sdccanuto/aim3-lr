package com.celebihacker.ml.logreg.mapred;

import java.io.IOException;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.mahout.common.IntPairWritable;

import com.celebihacker.ml.logreg.EnsembleJobTest;
import com.celebihacker.ml.util.AdaptiveLogger;

public class EvalReducer extends Reducer<Text, IntPairWritable, Text, Text> {
  
  private static AdaptiveLogger log = new AdaptiveLogger(
      EnsembleJobTest.RUN_LOCAL_MODE, Logger.getLogger(EvalReducer.class.getName()), Level.DEBUG);
  
  @Override
  public void reduce(Text key, Iterable<IntPairWritable> values, Context context) throws IOException, InterruptedException {
    log.debug("Eval Reducer for key: " + key.toString());
    long total=0;
    long correct=0;
    for (IntPairWritable pair : values) {
      total += pair.getFirst();
      correct += pair.getSecond();
      log.debug("- Total: " + pair.getFirst() + " Correct: " + pair.getSecond());
    }
    log.debug("- Accuracy: " + ((double)correct / (double)total));
    
    // TODO Emit results
  }

}
