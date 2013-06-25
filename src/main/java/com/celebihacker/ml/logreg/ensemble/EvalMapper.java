package com.celebihacker.ml.logreg.ensemble;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.mahout.common.IntPairWritable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.celebihacker.ml.RegressionModel;
import com.celebihacker.ml.logreg.LogisticRegressionEnsemble;
import com.celebihacker.ml.logreg.LogisticRegressionEnsemble.VotingSchema;
import com.celebihacker.ml.logreg.RCV1VectorReader;
import com.celebihacker.ml.util.AdaptiveLogger;
import com.celebihacker.ml.validation.OnlineAccuracy;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Closeables;

public class EvalMapper extends Mapper<Object, Text, Text, IntPairWritable> {
  
  private static AdaptiveLogger log = new AdaptiveLogger(
      EnsembleJob.RUN_LOCAL_MODE, Logger.getLogger(EvalMapper.class.getName()), Level.DEBUG); 
  
  private Vector y;
  
  private double THRESHOLD = 0.5;
  private Map<String, RegressionModel> models = Maps.newHashMap();
  private Map<String, OnlineAccuracy> accuracies = Maps.newHashMap();
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    log.debug("Eval Setup");
    // Read trained models from previous job output
    readEnsembleModels(context);

    // Read labels from distributed cache into memory (vector)
    Path[] files = DistributedCache.getLocalCacheFiles(context.getConfiguration());
    y = new DenseVector(EnsembleJob.TOTAL);
    RCV1VectorReader.readTarget(y, files[0].toString(), EnsembleJob.TARGET_POSITIVE);
  }
  
  private void readEnsembleModels(Context context) throws IOException {
    
    // TODO Make this generic for ensemble, global and majority. Build model classes for each with own prediction method
    Path dir = new Path(EnsembleJob.OUTPUT_PATH);
    FileSystem fs = FileSystem.get(context.getConfiguration());
    FileStatus[] statusList = fs.listStatus(dir, new PathFilter() {
      @Override
      public boolean accept(Path path) {
        if (path.getName().startsWith("part-r")) return true;
        else return false;
      }
    });

    ArrayList<RandomAccessSparseVector> ensembleModels = Lists.newArrayList();
    
    System.out.println("- Read trained models from " + statusList.length + " files");
    for (FileStatus status : statusList) {
      
//      BufferedReader reader = MLUtils.open(status.getPath().toString());
//      String line;
//      try {
//        while ((line = reader.readLine()) != null) {
//          // TODO Ensure that RandomAccessSparseVector is written in Mapper
//          VectorWritable ensembleModel = new VectorWritable();
//          Vector
//          ensembleModels.add((RandomAccessSparseVector)ensembleModel.get());
//          System.out.println("Partition " + partitionId.get() + ": Non zeros: " + ensembleModel.get().getNumNonZeroElements());
//        }
//      } finally {
//        Closeables.close(reader, true);
//      }
      
      
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, status.getPath(), context.getConfiguration());
      try {
        IntWritable partitionId = new IntWritable();
        VectorWritable ensembleModel = new VectorWritable();
        while (reader.next(partitionId, ensembleModel)) {
          ensembleModels.add((RandomAccessSparseVector)ensembleModel.get());
          System.out.println("- Ensemble-Model " + partitionId.get() + ": Non zeros: " + ensembleModel.get().getNumNonZeroElements());
        }
      } finally {
        Closeables.close(reader, true);
      }
    }
    
    models.put("ensemble-majority", new LogisticRegressionEnsemble(ensembleModels, THRESHOLD, VotingSchema.MAJORITY_VOTE));
    accuracies.put("ensemble-majority", new OnlineAccuracy(THRESHOLD));
    models.put("ensemble-merged", new LogisticRegressionEnsemble(ensembleModels, THRESHOLD, VotingSchema.MERGED_MODEL));
    accuracies.put("ensemble-merged", new OnlineAccuracy(THRESHOLD));
  }

  @Override
  public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
    // Read text into vector
    Vector v = new RandomAccessSparseVector(EnsembleJob.FEATURES);
    int docId = RCV1VectorReader.readRCV1Vector(v, value.toString());
    
    // Evaluate accuracy of all models
    for (Map.Entry<String, RegressionModel> model : models.entrySet()) {
      double prediction = model.getValue().predict(v);
      accuracies.get(model.getKey()).addSample((int)y.get(docId), prediction);
    }
  }
  
  @Override
  protected void cleanup(Context context)
      throws IOException, InterruptedException {
    log.debug("Cleanup: Write accuracy results");
    super.cleanup(context);
    // Write accuracy results
    for (Map.Entry<String, OnlineAccuracy> accuracy : accuracies.entrySet()) {
      context.write(
          new Text(accuracy.getKey()),
          new IntPairWritable(
              (int)accuracy.getValue().getTotal(),
              (int)accuracy.getValue().getCorrect()));
    }
  }
}