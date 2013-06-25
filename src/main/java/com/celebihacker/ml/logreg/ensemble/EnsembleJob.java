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
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.mahout.math.VectorWritable;

import com.celebihacker.ml.VectorLabeledWritable;

public class EnsembleJob extends Configured implements Tool {

  static final boolean RUN_LOCAL_MODE = true;
    
  // 47236 is highest term id
  static final int FEATURES = 47237;
  // 381327 points labeled with CCAT (RCV1-v2)
  // 810935 is highest document-id
  static final int TOTAL = 810935;
  static final int TARGETS = 2;
  static final String TARGET_POSITIVE = "CCAT";
  
  static final int REDUCE_TASKS = 4;
  
  private static String JOB_NAME = "aim3-ensemble-train";
  
  static final String LABEL_FILE_LOCAL = "/home/andre/dev/datasets/RCV1-v2/rcv1-v2.topics_ccat.qrels";
  static final String LABEL_FILE_HDFS = "rcv1-v2/rcv1-v2.topics_ccat.qrels";
  private static final String INPUT_FILE_LOCAL = "/home/andre/dev/datasets/RCV1-v2/vectors/lyrl2004_vectors_train.dat";
  private static final String INPUT_FILE_HDFS = "rcv1-v2/lyrl2004_vectors_train.dat";
  
  static final String JAR_PATH = "target/aim3-logreg-0.0.1-SNAPSHOT-job.jar";
  static final String CONFIG_FILE_PATH = "core-site.xml";
  
  static final String OUTPUT_PATH = "output-aim3-ensemble";
  
  /**
   * Will be called from ToolRunner internally
   * Hopefully passes us only the args after generic options
   */
  public int run(String[] args) throws Exception {

    Job job = prepareJob();

    // Broadcast Labels as a vector to all Reducers
    if (RUN_LOCAL_MODE) {
      DistributedCache.addCacheFile(new URI(LABEL_FILE_LOCAL), job.getConfiguration());
    } else {
      DistributedCache.addCacheFile(new URI(LABEL_FILE_HDFS), job.getConfiguration());
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
      new DeletingVisitor().accept(new File(OUTPUT_PATH));
      inputFile = INPUT_FILE_LOCAL;
    } else {
      System.out.println("RUN IN PSEUDO-DISTRIBUTED/CLUSTER MODE");
      inputFile = INPUT_FILE_HDFS;
      conf.addResource(new Path(CONFIG_FILE_PATH));
      
      // This jar has all required dependencies in it. Must be built first (mvn package)!
      conf.set("mapred.jar", JAR_PATH);
      
//      job.setNumReduceTasks(4);
      conf.setInt("mapred.reduce.tasks", REDUCE_TASKS);
      
      FileSystem hdfs = FileSystem.get(conf);
      Path path = new Path(OUTPUT_PATH);
      hdfs.delete(path, true);
    }
    System.out.println("Jar path: " + job.getJar());

    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(VectorLabeledWritable.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
  
    job.setMapperClass(EnsembleMapper.class);
//    job.setCombinerClass(Reduce.class);
    job.setReducerClass(EnsembleReducer.class);
  
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
  
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
