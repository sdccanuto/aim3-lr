package com.celebihacker.ml.logreg.mapred;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.net.URI;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.mahout.math.VectorWritable;

import com.celebihacker.ml.datasets.DatasetInfo;
import com.celebihacker.ml.datasets.RCV1DatasetInfo;
import com.celebihacker.ml.logreg.BatchGradientJobTest;
import com.celebihacker.ml.util.AdaptiveLogger;

import com.google.common.base.Joiner;

/**
 * Batch gradient for logistic regression
 */
public class BatchGradientJob extends Configured implements Tool {

  private static AdaptiveLogger LOGGER = new AdaptiveLogger(
      BatchGradientJobTest.RUN_LOCAL_MODE,
      Logger.getLogger(BatchGradientJob.class.getName()), Level.DEBUG);

  private static String JOB_NAME = "aim3-batch-gradient";

  static final int REDUCE_TASKS = 1;

  private String inputFileLocal;
  private String inputFileHdfs;
  private String jarPath;
  private String configFilePath;
  private String outputPath;
  private boolean runLocalMode;
  private final int maxIterations;

  public static final DatasetInfo rcv1 = RCV1DatasetInfo.get();
  private static final Joiner pathJoiner = Joiner.on("/");

  public BatchGradientJob(String inputFileLocal,
      String inputFileHdfs,
      String outputPath,
      String jarPath,
      String configFilePath,
      boolean runLocalMode,
      int maxIterations) {
    this.inputFileLocal = inputFileLocal;
    this.inputFileHdfs = inputFileHdfs;
    this.outputPath = outputPath;
    this.jarPath = jarPath;
    this.configFilePath = configFilePath;
    this.runLocalMode = runLocalMode;

    this.maxIterations = maxIterations;
  }

  @Override
  public int run(String[] args) throws Exception {

    boolean[] hasSucceeded = new boolean[this.maxIterations];

    // iterations
    for (int i = 0; i < this.maxIterations; i++) {
      LOGGER.debug("> starting iteration " + i);

      Job job = prepareJob();
      FileSystem fs = FileSystem.get(job.getConfiguration());

      // output path for this iteration
      Path iterationPath = new Path(pathJoiner.join(this.outputPath, "iteration" + i));
      FileOutputFormat.setOutputPath(job, iterationPath);

      if (i == 0) {

        // Remove data from previous runs
        if (this.runLocalMode) {
          new DeletingVisitor().accept(new File(this.outputPath));
        } else {
          fs.delete(new Path(this.outputPath), true);
        }

      } else {

        // Add weights of previous iteration to DistributedCache
        Path prevIterationPath = new Path(pathJoiner.join(this.outputPath, "iteration" + (i - 1)));
        LOGGER.debug("> output of previous iteration in " + prevIterationPath.toUri());

        FileStatus[] prevIterationWeights = fs.listStatus(prevIterationPath,
            new IterationOutputFilter());

        URI uri = prevIterationWeights[0].getPath().toUri();
        DistributedCache.addCacheFile(uri, job.getConfiguration());

        LOGGER.debug("> added " + uri + " to DistributedCache");
      }

      // execute job
      hasSucceeded[i] = job.waitForCompletion(true);
      LOGGER.debug("> completed iteration? " + hasSucceeded[i]);
    }

    for (int i = 0; i < this.maxIterations; i++) {
      if (!hasSucceeded[i])
        return 1;
    }

    return 0;
  }

  private Job prepareJob() throws IOException {

    System.out.println("-----------------");
    System.out.println("Prepare Job: " + JOB_NAME);
    System.out.println("-----------------");

    Job job = new Job(getConf(), JOB_NAME);
    Configuration conf = job.getConfiguration();
    job.setJarByClass(getClass());

    String inputFile = "";
    if (this.runLocalMode) {
      System.out.println("RUN IN LOCAL MODE");
      inputFile = this.inputFileLocal;
    } else {
      System.out.println("RUN IN PSEUDO-DISTRIBUTED/CLUSTER MODE");
      inputFile = this.inputFileHdfs;
      conf.addResource(new Path(this.configFilePath));

      // This jar has all required dependencies in it. Must be built first (mvn package)!
      conf.set("mapred.jar", this.jarPath);

      // job.setNumReduceTasks(4);
      conf.setInt("mapred.reduce.tasks", REDUCE_TASKS);
    }

    System.out.println("Jar path: " + job.getJar());

    job.setMapOutputKeyClass(NullWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);
    job.setOutputKeyClass(NullWritable.class);
    job.setOutputValueClass(VectorWritable.class);

    job.setMapperClass(GradientMapper.class);
    // job.setCombinerClass(GradientReducer.class);
    job.setReducerClass(GradientReducer.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    // configure the used input/output format class.
    FileInputFormat.addInputPath(job, new Path(inputFile));

    return job;
  }

  private static class IterationOutputFilter implements PathFilter {

    @Override
    public boolean accept(Path path) {
      if (path.getName().startsWith("part"))
        return true;

      return false;
    }
  }

  /**
   * Copied from MahoutTestCase. Recursively deletes folder and contained files
   */
  private static class DeletingVisitor implements FileFilter {

    @Override
    public boolean accept(File f) {
      if (!f.isFile()) {
        f.listFiles(this);
      }
      f.delete();
      return false;
    }
  }
}