package com.celebihacker.ml.logreg.ensemble;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.net.URI;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.mahout.common.IntPairWritable;

public class EvalJob extends Configured implements Tool {

  static final boolean RUN_LOCAL_MODE = true;
    
  static final int REDUCE_TASKS = 1;
  
  private static String JOB_NAME = "aim3-validation";
  
  private static final String INPUT_FILE_LOCAL = "/home/andre/dev/datasets/RCV1-v2/vectors/lyrl2004_vectors_test_pt0_5000.dat";
  private static final String INPUT_FILE_HDFS = "rcv1-v2/lyrl2004_vectors_train.dat";
  
  private static final String OUTPUT_PATH = "output-aim3";
  
  public int run(String[] args) throws Exception {

    Job job = prepareJob();

    // Broadcast Labels as a vector to all Reducers
    if (RUN_LOCAL_MODE) {
      DistributedCache.addCacheFile(new URI(EnsembleJob.LABEL_FILE_LOCAL), job.getConfiguration());
    } else {
      DistributedCache.addCacheFile(new URI(EnsembleJob.LABEL_FILE_HDFS), job.getConfiguration());
    }
    
    return job.waitForCompletion(true) ? 0 : 1;
  }
  
  private Job prepareJob() throws IOException {
    
    System.out.println("-----------------");
    System.out.println("Job: " + JOB_NAME);
    System.out.println("-----------------");
    
    Job job = new Job(getConf(), JOB_NAME);
    Configuration conf = job.getConfiguration();
    job.setJarByClass(getClass());
    
    String inputFile = "";
    if (RUN_LOCAL_MODE) {
      System.out.println("RUN IN LOCAL MODE");
      inputFile = INPUT_FILE_LOCAL;
      new DeletingVisitor().accept(new File(OUTPUT_PATH));
    } else {
      System.out.println("RUN IN PSEUDO-DISTRIBUTED/CLUSTER MODE");
      inputFile = INPUT_FILE_HDFS;
      conf.addResource(new Path(EnsembleJob.CONFIG_FILE_PATH));
      
      // This jar has all required dependencies in it. Must be built first (mvn package)!
      conf.set("mapred.jar", EnsembleJob.JAR_PATH);
      
      conf.setInt("mapred.reduce.tasks", REDUCE_TASKS);
      
      FileSystem hdfs = FileSystem.get(conf);
      Path path = new Path(OUTPUT_PATH);
      hdfs.delete(path, true);
    }
    System.out.println("Jar path: " + job.getJar());
    
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(IntPairWritable.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);
  
    job.setMapperClass(EvalMapper.class);
//    job.setCombinerClass(Reduce.class);
    job.setReducerClass(EvalReducer.class);
  
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(TextOutputFormat.class);
    
    // configure the used input/output format class.
    FileInputFormat.addInputPath(job, new Path(inputFile));
    FileOutputFormat.setOutputPath(job, new Path(OUTPUT_PATH));
    
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
