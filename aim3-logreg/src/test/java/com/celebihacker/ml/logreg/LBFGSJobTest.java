package com.celebihacker.ml.logreg;

import org.apache.hadoop.util.ToolRunner;
import org.junit.Test;

import com.celebihacker.ml.datasets.RCV1DatasetInfo;
import com.celebihacker.ml.logreg.mapred.GradientJob;
import com.celebihacker.ml.logreg.mapred.LBFGSJob;
import com.celebihacker.ml.logreg.mapred.TrainingErrorJob;

public class LBFGSJobTest {

  static final String INPUT_FILE_TRAIN_LOCAL = "/Users/uce/Desktop/rcv1-v2/vectors/lyrl2004_vectors_train_5000.seq";
//  static final String INPUT_FILE_TRAIN_HDFS = "hdfs://localhost:9000/user/uce/rcv1/rcv1-v2/lyrl2004_vectors_train_5000.seq";

  static final String OUTPUT_TRAIN_PATH = "output-aim3-batchgd";

  @Test
  public void test() throws Exception {

    TrainingErrorJob trainingErrorJob = new TrainingErrorJob(
        INPUT_FILE_TRAIN_LOCAL,
        OUTPUT_TRAIN_PATH);

    GradientJob gradientJob = new GradientJob(
        INPUT_FILE_TRAIN_LOCAL,
        OUTPUT_TRAIN_PATH);

    double eps = 0.1;

    double[] initial = new double[(int) RCV1DatasetInfo.get().getNumFeatures()];

    LBFGSJob lbfgs = new LBFGSJob(trainingErrorJob, gradientJob, eps, initial);

    ToolRunner.run(lbfgs, null);
  }
}