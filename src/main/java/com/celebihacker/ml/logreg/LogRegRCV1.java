package com.celebihacker.ml.logreg;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.regex.Pattern;

import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import com.celebihacker.ml.MLUtils;
import com.google.common.base.Splitter;

public class LogRegRCV1 {
  
  private static final int FEATURES = 90000;
  private static final int TARGETS = 2;
  
  private Splitter TRAC_SPLITTER = Splitter.on(Pattern.compile("[ :]"))
      .trimResults()
      .omitEmptyStrings();
    
  public void trainRCV1() throws IOException {
    
    // TRAIN
    OnlineLogisticRegression learningAlgorithm = new OnlineLogisticRegression(
    TARGETS, FEATURES, new L1())
    .alpha(1).stepOffset(1000)
    .decayExponent(0.2)
    .lambda(3.0e-5)
    .learningRate(20);
    // .alpha(1).stepOffset(1000)
    // decayExponent(0.9)
    // lambda: 3.0e-5
    // learningRate: 20
    
    // Read targets
    // CCAT(Corporate/Industrial)
    // ECAT(Economics)
    // GCAT(Government/Social)
    // MCAT(Markets)
    int totalItems = 900000;
    Vector yC = new DenseVector(totalItems);
    Vector yE = new DenseVector(totalItems);
    Vector yG = new DenseVector(totalItems);
    Vector yM = new DenseVector(totalItems);
    readTargets(yC, yE, yG, yM);
    
    // TODO Bug: Assumes that data are stored randomly in file
    // Line Format: doc-id  feature-id:val feature-id:val ...

    BufferedReader reader = MLUtils.open("/home/andre/dev/datasets/RCV1-v2/vectors/lyrl2004_vectors_train.dat");
    int lines = 0;
    int correct = 0;
    String line;
    while ((line = reader.readLine()) != null) {
      
      // Parse & transform to vector
      Vector v = new RandomAccessSparseVector(FEATURES);
      int docId = readRCV1Vector(v, line);
            
      // Test prediction
      int actualTarget = (int)yC.get(docId);
      double prediction = learningAlgorithm.classifyScalar(v);
      if (actualTarget == Math.round(prediction)) ++correct;
      
      // Train
      learningAlgorithm.train(actualTarget, v);
      
//      if (lines > 1000) break;
//      if (lines % 1000 == 0)
//        System.out.println(actualTarget + " - " + prediction);
      
      ++lines;
    }
    reader.close();
    System.out.println("Lines: " + lines);
    System.out.println("Correct: " + correct);
    System.out.println("Accuracy: " + ((double)correct)/((double)lines));
    
    learningAlgorithm.close();
    Matrix beta = learningAlgorithm.getBeta();
    System.out.println(beta.viewRow(0).viewPart(0, 100));
    
    System.out.println("----------------");
    System.out.println("TEST");
    System.out.println("----------------");
    
    reader = MLUtils.open("/home/andre/dev/datasets/RCV1-v2/vectors/lyrl2004_vectors_test_pt0.dat");
    lines = 0;
    correct = 0;
    while ((line = reader.readLine()) != null) {
      
      Vector v = new RandomAccessSparseVector(FEATURES);
      int docId = readRCV1Vector(v, line);
            
      int actualTarget = (int)yC.get(docId);
      double prediction = learningAlgorithm.classifyScalar(v);
      if (actualTarget == Math.round(prediction)) ++correct;
      
      // Train
      learningAlgorithm.train(actualTarget, v);
      
//      if (lines > 1000) break;
//      if (lines % 1000 == 0)
//        System.out.println(actualTarget + " - " + prediction);
      
      ++lines;
    }
    reader.close();
    System.out.println("Lines: " + lines);
    System.out.println("Correct: " + correct);
    System.out.println("Accuracy: " + ((double)correct)/((double)lines));
  }
  
  private int readRCV1Vector(Vector v, String line) {
    Iterator<String> iter = TRAC_SPLITTER.split(line).iterator();
    int docId = Integer.parseInt(iter.next());
    int featureId;
    Double featureVal;
    while (iter.hasNext()) {
      featureId = Integer.parseInt(iter.next());
      featureVal = Double.parseDouble(iter.next());
      v.set(featureId, featureVal);
    }
    return docId;
  }

  private void readTargets(Vector yC, Vector yE, Vector yG, Vector yM) throws IOException {
    // Line Format: ECAT 2286 1
    Splitter SPACE_SPLITTER = Splitter.on(Pattern.compile(" "))
        .trimResults()
        .omitEmptyStrings();
    BufferedReader reader = MLUtils.open("/home/andre/dev/datasets/RCV1-v2/rcv1-v2.topics.qrels");
    String line;
    String cat;
    while ((line = reader.readLine()) != null) {
      Iterator<String> iter = SPACE_SPLITTER.split(line).iterator();
      cat = iter.next();
      if (cat.equals("CCAT")) {
        yC.set(Integer.parseInt(iter.next()), 1d);
      }
      if (cat.equals("ECAT")) {
        yE.set(Integer.parseInt(iter.next()), 1d);
      }
      if (cat.equals("GCAT")) {
        yG.set(Integer.parseInt(iter.next()), 1d);
      }
      if (cat.equals("MCAT")) {
        yM.set(Integer.parseInt(iter.next()), 1d);
      }
    }
    reader.close();
  }
  
}
