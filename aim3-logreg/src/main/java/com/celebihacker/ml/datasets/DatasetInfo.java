package com.celebihacker.ml.datasets;

import java.util.List;
import java.util.Map;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.Lists;

/**
 * Provides additional information about a model (usually meta data)
 * E.g. mapping from feature dimension to feature name, number of features or size
 * 
 * For MapReduce we can make static instances of this class,
 *  that are accessible also from Mappers and Reducers
 * 
 */
public class DatasetInfo {
  
  private List<String> predictorNames = Lists.newArrayList();
//  private int numTargets;
  private long numFeatures;
  private long vectorSize;
  // TODO total is size of y vector for test and training -> no longer needed with sequence files!
  private long total;
  private BiMap<Integer, String> labelMap = HashBiMap.create();
  
  public DatasetInfo(List<String> predictorNames) {
    this.predictorNames = predictorNames;
    this.numFeatures = predictorNames.size();
    
    // TODO This is wrong, as there might be ids that are not in use
    this.vectorSize = predictorNames.size();
  }
  
  public DatasetInfo(long numFeatures,
      long vectorSize,
      long total,
      Map<Integer, String> labelMap) {
    this.vectorSize = vectorSize;
    this.numFeatures = numFeatures;
    this.total = total;
    this.labelMap = HashBiMap.create(labelMap);
  }
  
  public String getFeatureName(int dimension) {
    if (predictorNames.size()>dimension)
      return predictorNames.get(dimension);
    return "unknown feature";
  }
  
  public long getTotal() {
    return total;
  }
  
  public String getLabelNameById(int id) {
    return labelMap.get(id);
  }
  
  public int getLabelIdByName(String name) {
    return labelMap.inverse().get(name);
  }
  
  public long getNumFeatures() {
    return numFeatures;
  }
  
  public long getVectorSize() {
    return vectorSize;
  }
}
