package com.celebihacker.ml.logreg.mapred;

import java.io.File;
import java.io.FileFilter;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.celebihacker.ml.datasets.DatasetInfo;
import com.celebihacker.ml.datasets.RCV1DatasetInfo;
import com.celebihacker.ml.util.AdaptiveLogger;
import com.celebihacker.ml.util.HadoopUtils;
import com.google.common.base.Joiner;

/**
 * Batch gradient for logistic regression
 */
public class BatchGradientJob extends Configured implements Tool {

  private static AdaptiveLogger LOGGER = new AdaptiveLogger(
      Logger.getLogger(BatchGradientJob.class.getName()), 
      Level.DEBUG);

//  static final int REDUCE_TASKS = 1;

  private String inputFile;
  private String outputPath;
  private final int maxIterations;

  private final VectorWritable weights;

  public static final DatasetInfo rcv1 = RCV1DatasetInfo.get();
  private static final Joiner pathJoiner = Joiner.on("/");

  public BatchGradientJob(
      String inputFile,
      String outputPath,
      int maxIterations,
      double initial) {
    this.inputFile = inputFile;
    this.outputPath = outputPath;

    this.maxIterations = maxIterations;

    Vector vec = new SequentialAccessSparseVector((int) rcv1.getNumFeatures());
    
    vec.assign(initial);

    this.weights = new VectorWritable(vec);
  }
  
  @Override
  public int run(String[] args) throws Exception {

    // Non zero numbers for rcv1-v2 (5000): 21871 -> 19199 -> 19165
    
    boolean[] hasSucceeded = new boolean[this.maxIterations];
    
    // Configuration object for file system actions
    Configuration conf = new Configuration();
    conf.addResource(new Path(GlobalJobSettings.CONFIG_FILE_PATH));
    boolean runLocal = HadoopUtils.detectLocalMode(conf);

    // iterations
    for (int i = 0; i < this.maxIterations; i++) {
      LOGGER.debug("> starting iteration " + i);
      
      // output path for this iteration
      Path iterationPath = new Path(pathJoiner.join(this.outputPath, "iteration" + i));

//      Job job = prepareJob();
      FileSystem fs = FileSystem.get(conf);
//      FileOutputFormat.setOutputPath(job, iterationPath);
      
      GradientJob job = new GradientJob(
          inputFile,
          iterationPath.toString());
      
      if (i == 0) {

        // Remove data from previous runs (delete root output folder recursively)
        if (runLocal) {
          new DeletingVisitor().accept(new File(this.outputPath));
        } else {
          fs.delete(new Path(this.outputPath), true);
        }
        
        // Initial weights
//        Path cachePath = new Path(conf.get("hadoop.tmp.dir") + "/initial_weights");
//        HadoopUtils.writeVectorToDistCache(conf, this.weights, cachePath);
//        LOGGER.debug("> added " + cachePath.toUri() + " to DistributedCache");
        
      } else {

        // Add weights of previous iteration to DistributedCache (existing file)
        Path prevIterationPath = new Path(pathJoiner.join(this.outputPath, "iteration" + (i - 1)));
        FileStatus[] prevIterationWeights = fs.listStatus(prevIterationPath, new IterationOutputFilter());
        
        Path cachePath = prevIterationWeights[0].getPath();
        this.weights.set(readVectorFromHDFS(cachePath, conf));
//        DistributedCache.addCacheFile(cachePath.toUri(), conf);
//        LOGGER.debug("> added " + cachePath.toUri() + " to DistributedCache");

      }
      
      // GradientJob will write this vector to distributed cache
      job.setWeightVector(this.weights.get());
      
      // execute job
//      hasSucceeded[i] = job.waitForCompletion(false);
      hasSucceeded[i] = (ToolRunner.run(job, null)==0) ? true : false;
      LOGGER.debug("> completed iteration? " + hasSucceeded[i]);
    }

    for (int i = 0; i < this.maxIterations; i++) {
      if (!hasSucceeded[i])
        return 1;
    }

    return 0;
  }
  
  private Vector readVectorFromHDFS(Path filePath, Configuration conf) {
    Vector w = null;
    for (Pair<NullWritable, VectorWritable> weights : new SequenceFileIterable<NullWritable, VectorWritable>(
        filePath, conf)) {
      w = weights.getSecond().get();
      System.out.println("Read from distributed cache in gradient mapper");
      System.out.println("- non zeros: " + w.getNumNonZeroElements());
    }
    return w;
  }

//  private Job prepareJob() throws IOException {
//
//    System.out.println("-----------------");
//    System.out.println("Prepare Job: " + JOB_NAME);
//    System.out.println("-----------------");
//
//    Job job = new Job(getConf(), JOB_NAME);
//    Configuration conf = job.getConfiguration();
//    job.setJarByClass(getClass());
//
//    String inputFile = "";
//    boolean runLocal = HadoopUtils.detectLocalMode(conf);
//    if (runLocal) {
//      System.out.println("RUN IN LOCAL MODE");
//      inputFile = this.inputFileLocal;
//    } else {
//      System.out.println("RUN IN PSEUDO-DISTRIBUTED/CLUSTER MODE");
//      inputFile = this.inputFileHdfs;
//      conf.addResource(new Path(this.configFilePath));
//
//      // This jar has all required dependencies in it. Must be built first (mvn package)!
//      conf.set("mapred.jar", this.jarPath);
//
//      // job.setNumReduceTasks(4);
//      conf.setInt("mapred.reduce.tasks", REDUCE_TASKS);
//    }
//
//    System.out.println("Jar path: " + job.getJar());
//
//    job.setMapOutputKeyClass(NullWritable.class);
//    job.setMapOutputValueClass(VectorWritable.class);
//    job.setOutputKeyClass(NullWritable.class);
//    job.setOutputValueClass(VectorWritable.class);
//
//    job.setMapperClass(GradientMapper.class);
//    // job.setCombinerClass(GradientReducer.class);
//    job.setReducerClass(GradientReducer.class);
//
//    job.setInputFormatClass(SequenceFileInputFormat.class);
//    job.setOutputFormatClass(SequenceFileOutputFormat.class);
//
//    // configure the used input/output format class.
//    FileInputFormat.addInputPath(job, new Path(inputFile));
//
//    return job;
//  }

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