package com.celebihacker.ml.logreg;

import org.apache.hadoop.util.ToolRunner;
import org.junit.Test;

import com.celebihacker.ml.datasets.RCV1DatasetInfo;
import com.celebihacker.ml.logreg.mapred.GradientJob;
import com.celebihacker.ml.logreg.mapred.LBFGSJob;
import com.celebihacker.ml.logreg.mapred.TrainingErrorJob;

public class LBFGSJobTest {

  public static final boolean RUN_LOCAL_MODE = true;

  static final String INPUT_FILE_TRAIN_LOCAL = "/Users/uce/Desktop/rcv1-v2/vectors/lyrl2004_vectors_train_5000.seq";
  static final String INPUT_FILE_TRAIN_HDFS = "hdfs://localhost:9000/user/uce/rcv1/rcv1-v2/lyrl2004_vectors_train_5000.seq";

  static final String OUTPUT_TRAIN_PATH = "output-aim3-batchgd";

  static final String JAR_PATH = "target/aim3-logreg-0.0.1-SNAPSHOT-job.jar";
  static final String CONFIG_FILE_PATH = "core-site.xml";

  @Test
  public void test() throws Exception {

    TrainingErrorJob trainingErrorJob = new TrainingErrorJob(
        INPUT_FILE_TRAIN_LOCAL,
        INPUT_FILE_TRAIN_HDFS,
        OUTPUT_TRAIN_PATH,
        JAR_PATH,
        CONFIG_FILE_PATH,
        RUN_LOCAL_MODE);

    GradientJob gradientJob = new GradientJob(
        INPUT_FILE_TRAIN_LOCAL,
        INPUT_FILE_TRAIN_HDFS,
        OUTPUT_TRAIN_PATH,
        JAR_PATH,
        CONFIG_FILE_PATH,
        RUN_LOCAL_MODE);

    double eps = 0.1;
    int maxIterations = 5;

    double[] initial = new double[(int) RCV1DatasetInfo.get().getNumFeatures()];

    LBFGSJob lbfgs = new LBFGSJob(trainingErrorJob, gradientJob, eps, maxIterations, initial);

    ToolRunner.run(lbfgs, null);
  }
}