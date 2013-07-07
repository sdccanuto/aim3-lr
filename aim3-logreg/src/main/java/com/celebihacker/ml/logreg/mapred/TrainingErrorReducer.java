package com.celebihacker.ml.logreg.mapred;

import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import com.celebihacker.ml.logreg.EnsembleJobTest;
import com.celebihacker.ml.util.AdaptiveLogger;

public class TrainingErrorReducer extends
    Reducer<NullWritable, DoubleWritable, NullWritable, DoubleWritable> {

  private static AdaptiveLogger LOGGER = new AdaptiveLogger(
      EnsembleJobTest.RUN_LOCAL_MODE, Logger.getLogger(TrainingErrorReducer.class.getName()),
      Level.DEBUG);

  @Override
  public void reduce(NullWritable key, Iterable<DoubleWritable> values, Context context)
      throws IOException, InterruptedException {

    double trainingError = 0.0;

    for (DoubleWritable partialTrainingError : values) {
      trainingError += partialTrainingError.get();
    }

    LOGGER.debug("TrainingErrorReducer: training error of " + trainingError);

    context.write(NullWritable.get(), new DoubleWritable(trainingError));
  }
}