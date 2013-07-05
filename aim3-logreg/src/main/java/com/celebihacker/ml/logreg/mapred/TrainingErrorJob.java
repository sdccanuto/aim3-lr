package com.celebihacker.ml.logreg.mapred;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.celebihacker.ml.datasets.RCV1DatasetInfo;
import com.google.common.base.Joiner;

/**
 * Computes the in-sample error for logistic regression
 */
public class TrainingErrorJob extends Configured implements Tool {

  private static String JOB_NAME = "aim3-training-error";

  static final int REDUCE_TASKS = 1;

  private String inputFileLocal;
  private String inputFileHdfs;
  private String jarPath;
  private String configFilePath;
  private String outputPath;

  private boolean runLocalMode;
  private final VectorWritable weights;

  private static final Joiner pathJoiner = Joiner.on("/");

  public TrainingErrorJob(String inputFileLocal,
      String inputFileHdfs,
      String outputPath,
      String jarPath,
      String configFilePath,
      boolean runLocalMode) {
    this.inputFileLocal = inputFileLocal;
    this.inputFileHdfs = inputFileHdfs;
    this.outputPath = outputPath;
    this.jarPath = jarPath;
    this.configFilePath = configFilePath;
    this.runLocalMode = runLocalMode;

    Vector vec = new SequentialAccessSparseVector((int) RCV1DatasetInfo.get().getNumFeatures());
    this.weights = new VectorWritable(vec);
  }

  @Override
  public int run(String[] args) throws Exception {

    Job job = prepareJob();

    // Initial weights
    Configuration conf = new Configuration();
    Path toCache = new Path(pathJoiner.join(conf.get("hadoop.tmp.dir"), "initial_weights"));

    SequenceFile.Writer writer = SequenceFile.createWriter(FileSystem.getLocal(conf), conf,
        toCache, NullWritable.class, VectorWritable.class);
    
    writer.append(NullWritable.get(), this.weights);
    writer.close();

    // Add to distributed cache
    DistributedCache.addCacheFile(toCache.toUri(), job.getConfiguration());

    return job.waitForCompletion(true) ? 0 : 1;
  }

  public void setWeightVector(double initial) {
    this.weights.get().assign(initial);
  }

  public void setWeightVector(double[] weights) {
    this.weights.get().assign(weights);
  }

  public void setWeightVector(Vector weights) {
    this.weights.get().assign(weights);
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
      new DeletingVisitor().accept(new File(this.outputPath));
      inputFile = this.inputFileLocal;
    } else {
      System.out.println("RUN IN PSEUDO-DISTRIBUTED/CLUSTER MODE");
      inputFile = this.inputFileHdfs;
      conf.addResource(new Path(this.configFilePath));

      // This jar has all required dependencies in it. Must be built first (mvn package)!
      conf.set("mapred.jar", this.jarPath);

      // job.setNumReduceTasks(4);
      conf.setInt("mapred.reduce.tasks", REDUCE_TASKS);

      FileSystem hdfs = FileSystem.get(conf);
      Path path = new Path(this.outputPath);
      hdfs.delete(path, true);
    }
    System.out.println("Jar path: " + job.getJar());

    job.setMapOutputKeyClass(NullWritable.class);
    job.setMapOutputValueClass(DoubleWritable.class);
    job.setOutputKeyClass(NullWritable.class);
    job.setOutputValueClass(DoubleWritable.class);

    job.setMapperClass(TrainingErrorMapper.class);
    // job.setCombinerClass(GradientReducer.class);
    job.setCombinerClass(TrainingErrorReducer.class);
    job.setReducerClass(TrainingErrorReducer.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    // configure the used input/output format class.
    FileInputFormat.addInputPath(job, new Path(inputFile));
    FileOutputFormat.setOutputPath(job, new Path(this.outputPath));

    return job;
  }
  
  public String getOutputPath() {
    return this.outputPath;
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