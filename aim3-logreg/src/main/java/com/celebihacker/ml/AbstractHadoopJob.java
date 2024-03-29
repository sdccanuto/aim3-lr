package com.celebihacker.ml;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.OutputFormat;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;

import com.celebihacker.ml.util.HadoopUtils;
import com.celebihacker.ml.util.IOUtils;

public abstract class AbstractHadoopJob extends Configured implements Tool {
  
  protected Job prepareJob(
      String jobName,
      int numReducers,
      @SuppressWarnings("rawtypes") Class<? extends Mapper> mapper,
      @SuppressWarnings("rawtypes") Class<? extends Reducer> reducer,
      Class<? extends Writable> mapOutputKey,
      Class<? extends Writable> mapOutputValue,
      Class<? extends Writable> reduceOutputKey,
      Class<? extends Writable> reduceOutputValue,
      @SuppressWarnings("rawtypes") Class<? extends InputFormat> inputFormat,
      @SuppressWarnings("rawtypes") Class<? extends OutputFormat> outputFormat,
      String inputPath,
      String outputPath
      ) throws IOException {
    
    System.out.println("-----------------");
    System.out.println("Prepare Job: " + jobName);
    System.out.println("-----------------");
    
    Job job = new Job(getConf(), jobName);
    Configuration conf = job.getConfiguration();
    setConf(conf);
    System.out.println("EQUALITY? " + getConf().equals(conf));

    conf.addResource(new Path(GlobalSettings.CONFIG_FILE_PATH));
    
    boolean runLocal = HadoopUtils.detectLocalMode(conf);
    if (runLocal) {
      System.out.println("RUN IN LOCAL MODE");
    } else {
      System.out.println("RUN IN PSEUDO-DISTRIBUTED/CLUSTER MODE");
    }
    
    if (reducer.equals(Reducer.class)) {
      if (mapper.equals(Mapper.class)) {
        throw new IllegalStateException("Can't figure out the user class jar file from mapper/reducer");
      }
      job.setJarByClass(mapper);
    } else {
      job.setJarByClass(reducer);
    }
    if (!runLocal) {
      // This is needed if we run from eclipse, which won't build a jar automatically
      // In this case we have to build the jar manually before!
      // mvn package will build a jar with all required dependencies
      conf.set("mapred.jar", GlobalSettings.JAR_PATH);
      
//      job.setNumReduceTasks(4);
      conf.setInt("mapred.reduce.tasks", numReducers);
    }
    System.out.println("Jar path: " + job.getJar());
    
    // ----- Set classes -----

    job.setMapperClass(mapper);
    job.setReducerClass(reducer);
    
    job.setMapOutputKeyClass(mapOutputKey);
    job.setMapOutputValueClass(mapOutputValue);
    job.setOutputKeyClass(reduceOutputKey);
    job.setOutputValueClass(reduceOutputValue);
    
    job.setInputFormatClass(inputFormat);
    job.setOutputFormatClass(outputFormat);
    
    // ----- Input / Output path -----
    
    FileInputFormat.addInputPath(job, new Path(inputPath));
    FileOutputFormat.setOutputPath(job, new Path(outputPath));
    
    return job;
  }
  
  protected void cleanupOutputDirectory(String outputPath) throws IOException {
    boolean runLocal = HadoopUtils.detectLocalMode(getConf());
    if (runLocal) {
      IOUtils.deleteRecursively(outputPath);
    } else {
      FileSystem fs = FileSystem.get(getConf());
      Path path = new Path(outputPath);
      fs.delete(path, true);
    }
  }

}
