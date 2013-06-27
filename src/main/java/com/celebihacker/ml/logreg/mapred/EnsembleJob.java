package com.celebihacker.ml.logreg.mapred;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.mahout.math.VectorWritable;

import com.celebihacker.ml.DatasetInfo;
import com.celebihacker.ml.VectorMultiLabeledWritable;
import com.celebihacker.ml.logreg.datasets.RCV1DatasetInfo;

public class EnsembleJob extends Configured implements Tool {

  private static String JOB_NAME = "aim3-ensemble-train";
  
  private String inputFileLocal;
  private String inputFileHdfs;
  private String jarPath;
  private String configFilePath;
  private String outputPath;
  private boolean runLocalMode;
  private int partitions;
  
  static DatasetInfo datasetInfo = RCV1DatasetInfo.get();
  
  // TODO Feature: Currently we train a hardcoded single 1-vs-all classifier
  static final String TARGET_POSITIVE = "CCAT";
  
  public EnsembleJob(String inputFileLocal,
      String inputFileHdfs,
      String outputPath,
      String jarPath,
      String configFilePath,
      boolean runLocalMode,
      int partitions) {
    this.inputFileLocal = inputFileLocal;
    this.inputFileHdfs = inputFileHdfs;
    this.outputPath = outputPath;
    this.jarPath = jarPath;
    this.configFilePath = configFilePath;
    this.runLocalMode = runLocalMode;
    this.partitions = partitions;
  }

  public int run(String[] args) throws Exception {

    Job job = prepareJob();
    
    return job.waitForCompletion(true) ? 0 : 1;
  }
  
  private Job prepareJob() throws IOException {
    
    System.out.println("-----------------");
    System.out.println("Prepare Job: " + JOB_NAME);
    System.out.println("-----------------");
    
    Job job = new Job(getConf(), JOB_NAME);
    Configuration conf = job.getConfiguration();
    job.setJarByClass(getClass());
    
    String inputFile = "";
    if (runLocalMode) {
      System.out.println("RUN IN LOCAL MODE");
      new DeletingVisitor().accept(new File(outputPath));
      inputFile = inputFileLocal;
    } else {
      System.out.println("RUN IN PSEUDO-DISTRIBUTED/CLUSTER MODE");
      inputFile = inputFileHdfs;
      conf.addResource(new Path(configFilePath));
      
      // This jar has all required dependencies in it. Must be built first (mvn package)!
      conf.set("mapred.jar", jarPath);
      
//      job.setNumReduceTasks(4);
      conf.setInt("mapred.reduce.tasks", partitions);
      
      FileSystem hdfs = FileSystem.get(conf);
      Path path = new Path(outputPath);
      hdfs.delete(path, true);
    }
    System.out.println("Jar path: " + job.getJar());

    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(VectorMultiLabeledWritable.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
  
    job.setMapperClass(EnsembleMapper.class);
//    job.setCombinerClass(Reduce.class);
    job.setReducerClass(EnsembleReducer.class);
  
//    job.setInputFormatClass(TextInputFormat.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
  
    // configure the used input/output format class.
    FileInputFormat.addInputPath(job, new Path(inputFile));
    FileOutputFormat.setOutputPath(job, new Path(outputPath));
    
    return job;
  }

  /**
   * Copied from MahoutTestCase. Recursively deletes folder and contained files
   */
  private static class DeletingVisitor implements FileFilter {
    
    public boolean accept(File f) {
      if (!f.isFile()) {
        f.listFiles(this);
      }
      f.delete();
      return false;
    }
  }

}
