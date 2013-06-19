package com.celebihacker.ml.logreg.ensemble;

import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import com.celebihacker.ml.AdaptiveLogger;
import com.celebihacker.ml.VectorLabeledWritable;
import com.celebihacker.ml.logreg.RCV1VectorReader;

public class EnsembleMapper extends Mapper<Object, Text, IntWritable, VectorLabeledWritable> {
  
  Random random = new Random();

  int numberReducers;

  IntWritable curPartition = new IntWritable();
  
  private static AdaptiveLogger log = new AdaptiveLogger(
      EnsembleJob.RUN_LOCAL_MODE, Logger.getLogger(EnsembleMapper.class.getName())); 
  
  private Vector y;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);

    log.setLevel(Level.DEBUG);
    
    numberReducers = Integer.parseInt(context.getConfiguration().get("mapred.reduce.tasks"));
    log.debug("Number reducers: " + numberReducers);

    // Read labels from distributed cache into memory (vector)
    Path[] files = DistributedCache.getLocalCacheFiles(context.getConfiguration());
    y = new DenseVector(EnsembleJob.TOTAL);
    RCV1VectorReader.readTarget(y, files[0].toString(), EnsembleJob.TARGET_POSITIVE);
  }

  private static VectorLabeledWritable currentLabeledVector = new VectorLabeledWritable();
  
  @Override
  public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
    
    // Read text into vector
    Vector v = new RandomAccessSparseVector(EnsembleJob.FEATURES);
    int docId = RCV1VectorReader.readRCV1Vector(v, value.toString());

    // Randomly distribute to Reducers to get a random partitioning
    
    // TODO If Reducer uses SGD (Online, one-pass), we should also randomize the order within each partition!?
    
    // TODO Bug: Add custom partitioner to make sure that different partitions are sent to different reducers 
    
    curPartition.set(random.nextInt(numberReducers));
    currentLabeledVector.setVector(v);
    currentLabeledVector.setLabel((int)y.get(docId));
    context.write(curPartition, currentLabeledVector);
  }
}