package com.celebihacker.ml.logreg;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VectorWritable;

/**
 * TODO Improvement: Switch to current stable Release 1.1.2 (currently using 1.0.4) 
 *
 */
public class RCV1Job extends Configured implements Tool {
    
  public static class Map extends Mapper<Object, Text, IntWritable, VectorWritable> {
    
    Random random = new Random();
    int numberReducers = 2;
    IntWritable partition = new IntWritable();
    private static Logger logger = Logger.getLogger(Map.class.getName()); 

    @Override
    protected void setup(org.apache.hadoop.mapreduce.Mapper.Context context)
        throws IOException, InterruptedException {
      super.setup(context);
      
      logger.setLevel(Level.DEBUG);
      numberReducers = Integer.parseInt(context.getConfiguration().get("mapred.reduce.tasks"));
      System.out.println("Number reducers: " + numberReducers);
    }
    
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      // Randomly distribute to Reducers to get a random partitioning
      // TODO: If Reducer uses SGD (Online, one-pass), we should also randomize the order within each partition?!
      logger.debug(key + " -> " + partition.get());
      partition.set(random.nextInt(numberReducers));
      context.write(partition, new VectorWritable(new RandomAccessSparseVector(2)));
    }
  }

  /*
    class for Reduce
  */
  public static class Reduce extends Reducer<IntWritable, VectorWritable, IntWritable, Text> {
    
    Text currentOutput = new Text();
    private static Logger logger = Logger.getLogger(Reduce.class.getName());
    
    @Override
    protected void setup(org.apache.hadoop.mapreduce.Reducer.Context context)
        throws IOException, InterruptedException {
      super.setup(context);
      logger.setLevel(Level.DEBUG);
    }
  
    public void reduce(IntWritable key, Iterable<VectorWritable> values, Context context) throws IOException, InterruptedException {
      // Train a full model, using sequential in-memory technique
      for (VectorWritable val : values) {
        currentOutput.set(val.toString());
        context.write(key, currentOutput);
      }
    }

  }
  
  public int run(String[] args) throws Exception {
//    if (args.length != 2) {
//      System.err.printf("Usage: %s [generic options] <input> <output>", getClass().getSimpleName());
//      ToolRunner.printGenericCommandUsage(System.err);
//      return -1;
//    }
//    String inputFile = args[0];
//    String outputDir = args[1];

    Job job = prepareJob();
  
    return job.waitForCompletion(true) ? 0 : 1;
  }
  
  private Job prepareJob() throws IOException {

    Job job = new Job(getConf(), "rcv1");
    Configuration conf = job.getConfiguration();
    job.setJarByClass(getClass());

    String inputFile = "";  // will be set depending on localmode or not
    String outputDir = "output-aim3";
    boolean runLocalMode = false;
    
    // Work with local information (not in local mode)
    // Requires building of the jar first: mvn package
    if (runLocalMode) {
      new DeletingVisitor().accept(new File(outputDir));
      inputFile = "/home/andre/dev/datasets/RCV1-v2/vectors/lyrl2004_vectors_train_5000.dat";
    } else {
      // reads configuration in core-site.xml. hdfs for me
      inputFile = "rcv1-v2/lyrl2004_vectors_train_5000.dat";
      conf.addResource(new Path("core-site.xml"));
//      conf.set("mapred.jar","target/aim3-logreg-0.0.1-SNAPSHOT.jar");
      conf.set("mapred.jar","target/aim3-logreg-0.0.1-SNAPSHOT-job.jar");
      
      
      // Delete old output dir
      FileSystem hdfs = FileSystem.get(conf);
      Path path = new Path(outputDir);
      hdfs.delete(path, true);
    }
    System.out.println("Jar: " + job.getJar());
    
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
  
    job.setMapperClass(Map.class);
//    job.setCombinerClass(Reduce.class);
    job.setReducerClass(Reduce.class);
  
    // Input/Output Format
    // We use the default.
    // For file based input/output formats we can use the predefined FileInputFormat class
    // It has a generic implementation of getSplits()
    // One map tasks is spawned for each InputSplit generated by the InputFormat
    // see http://hadoop.apache.org/docs/current/api/org/apache/hadoop/mapred/InputFormat.html
    //job.setInputFormatClass(TextInputFormat.class);
    //job.setOutputFormatClass(TextOutputFormat.class);
  
    // configure the used input/output format class.
    FileInputFormat.addInputPath(job, new Path(inputFile));
    FileOutputFormat.setOutputPath(job, new Path(outputDir));
  
    // Set number of mappers and reducers manually
    // Not done here, will be done via command line parameter
    //conf.setNumMapTasks(4);
//    job.setNumReduceTasks(4);
    conf.setInt("mapred.reduce.tasks", 4);
//    conf.setInt("mapred.tasktracker.reduce.tasks.maximum", 4);
    
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
